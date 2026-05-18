# examples/05_forward_prediction.R
#
# Forward genomic prediction with BayesA + BayesR. Uses the multi-generation
# extension of the bundled demo dataset (load_data("small")$multigen):
#
#   Scenario A: train at gen-1,         predict gen-2
#   Scenario B: train at gen-1 + gen-2, predict gen-3
#
# Architecture fixed at QTL@MH. Two marker bases (SNP, MH) and two models
# (BayesA, BayesR) -> 8 fits. Reports r(GEBV, TBV) with bootstrap SE,
# rendered as a ggplot bar chart.
#
# Requirements: masbayes, ggplot2, dplyr
# Usage: source("masbayes/examples/05_forward_prediction.R")

suppressPackageStartupMessages({
  library(masbayes); library(ggplot2); library(dplyr)
})

# Resolve the directory this script lives in, so plots save next to the
# script regardless of the caller's working directory. Works for both
# `Rscript path/to/file.R` and interactive `source("path/to/file.R")`.
script_dir <- local({
  fa <- grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(fa) > 0) {
    return(dirname(normalizePath(sub("--file=", "", fa[1]))))
  }
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) dirname(normalizePath(ofile)) else getwd()
})

d   <- load_data("small")$multigen
bid <- attr(d$gen1$mh, "block_id")
MCMC_P <- list(n_iter = 2000L, n_burn = 1000L, n_thin = 5L, seed = 123L)
N_BOOT <- 200L

# Build the train + test design + target vectors for one (scenario, marker)
# cell. Architecture is always QTL@MH (the canonical multi-allelic case).
# Returns enough state for any model to fit + predict + evaluate.
build_pack <- function(scenario, marker) {
  if (scenario == "A") {
    train_pheno <- d$gen1$pheno
    test_cohort <- d$gen2_mh
    train_snp   <- d$gen1$snp
    ref_struct  <- d$reference_structure_gen1
  } else {
    train_pheno <- rbind(
      d$gen1$pheno[,    c("id", "sex", "y_cont_qtl_mh")],
      d$gen2_mh$pheno[, c("id", "sex", "y_cont_qtl_mh")])
    test_cohort <- d$gen3_mh
    train_snp   <- rbind(d$gen1$snp, d$gen2_mh$snp)
    ref_struct  <- d$reference_structure_gen1_gen2_mh
  }

  if (marker == "snp") {
    snp_tr <- construct_snp_matrix(train_snp)
    W_tr <- snp_tr$W
    W_te <- construct_snp_matrix(test_cohort$snp, ref_freq = snp_tr$freq)$W
    mtype <- "snp"
  } else {
    W_tr <- ref_struct$W_ah
    W_te <- construct_wah_matrix(test_cohort$mh, bid, NULL,
                                  reference_structure = ref_struct)$W_ah
    mtype <- "multiallelic"
  }

  X_tr <- model.matrix(~ train_pheno$sex - 1)
  X_te <- model.matrix(~ test_cohort$pheno$sex - 1)
  colnames(X_tr) <- colnames(X_te) <- c("F", "M")

  list(y_tr = train_pheno$y_cont_qtl_mh, W_tr = W_tr, X_tr = X_tr,
       W_te = W_te, X_te = X_te,
       tbv_te = test_cohort$pheno$tbv_qtl_mh_true,
       y_te   = test_cohort$pheno$y_cont_qtl_mh,
       marker_type = mtype,
       wtw_diag = colSums(W_tr ^ 2),
       sigma2_init = var(train_pheno$y_cont_qtl_mh) * 0.5)
}

fit_one <- function(model_name, p) {
  if (model_name == "bayesa") {
    run_bayesa(w = p$W_tr, X = p$X_tr, y = p$y_tr, wtw_diag = p$wtw_diag,
               marker_type = p$marker_type,
               nu = 4.5, sigma2_g = p$sigma2_init,
               sigma2_e_init = p$sigma2_init,
               prior_params = list(a0_e = 10), mcmc_params = MCMC_P,
               method = "mcmc", save_rds = FALSE, verbose = FALSE)
  } else {
    run_bayesr(w = p$W_tr, X = p$X_tr, y = p$y_tr, wtw_diag = p$wtw_diag,
               marker_type = p$marker_type,
               pi_vec = c(0.90, 0.05, 0.03, 0.02),
               sigma2_e_init = p$sigma2_init, sigma2_ah = p$sigma2_init,
               prior_params = list(a0_e = 10, a0_g = 10,
                                   variance_class = c(0, 0.01, 0.1, 1)),
               mcmc_params = MCMC_P,
               method = "mcmc", save_rds = FALSE, verbose = FALSE)
  }
}

boot_se <- function(gebv, tbv, n_boot, seed) {
  set.seed(seed)
  rs <- replicate(n_boot, {
    idx <- sample(seq_along(tbv), replace = TRUE)
    suppressWarnings(cor(gebv[idx], tbv[idx]))
  })
  sd(rs, na.rm = TRUE)
}

# Build each (scenario, marker) pack ONCE and reuse for both BayesA + BayesR.
sink_path <- tempfile()
res <- expand.grid(scenario = c("A", "B"), marker = c("snp", "mh"),
                   model = c("bayesa", "bayesr"),
                   r_test_g = NA_real_, se = NA_real_,
                   stringsAsFactors = FALSE)

packs <- list()
for (sc in c("A", "B")) for (mk in c("snp", "mh")) {
  sink(sink_path)
  packs[[paste(sc, mk, sep = ".")]] <- build_pack(sc, mk)
  sink()
}

for (i in seq_len(nrow(res))) {
  pack <- packs[[paste(res$scenario[i], res$marker[i], sep = ".")]]
  sink(sink_path)
  fit  <- fit_one(res$model[i], pack)
  pred <- predict(fit, pack$W_te, pack$y_te, X_new = pack$X_te)
  sink()
  res$r_test_g[i] <- cor(pred$GEBV, pack$tbv_te)
  res$se[i]       <- boot_se(pred$GEBV, pack$tbv_te, N_BOOT, seed = 1000L + i)
}
unlink(sink_path)

# ── Accuracy table ────────────────────────────────────────────────────────
res_print <- res %>%
  mutate(scenario = ifelse(scenario == "A",
                           "A (gen1->gen2)", "B (gen1+2->gen3)")) %>%
  arrange(scenario, marker, model)
cat("\n=== Forward prediction r_test_g (QTL@MH architecture) ===\n\n")
print(res_print, row.names = FALSE)

# ── Bar plot with SE ──────────────────────────────────────────────────────
plot_df <- res %>%
  mutate(scenario = factor(scenario, levels = c("A", "B"),
                           labels = c("A: gen1 -> gen2",
                                      "B: gen1+gen2 -> gen3")),
         marker = toupper(marker),
         model  = factor(model, levels = c("bayesa", "bayesr"),
                         labels = c("BayesA", "BayesR")))

# Forward prediction accuracy under the QTL@MH architecture: BayesA vs BayesR
# on SNP and MH markers, faceted by training scenario. Error bars are bootstrap
# SE (200 resamples on the test set, single seed).
p <- ggplot(plot_df, aes(marker, r_test_g, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = r_test_g - se, ymax = r_test_g + se),
                position = position_dodge(width = 0.8), width = 0.2) +
  facet_wrap(~ scenario) +
  scale_fill_manual(values = c(BayesA = "#4C72B0", BayesR = "#DD8452")) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(x = "Marker", y = "r(GEBV, TBV)  ± bootstrap SE", fill = NULL) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        panel.grid.major.x = element_blank())

print(p)
ggsave(file.path(script_dir, "05_forward_prediction.png"),
       p, width = 6, height = 4, dpi = 300, bg = "white")

invisible(list(results = res, plot = p))
