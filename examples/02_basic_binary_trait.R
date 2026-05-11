# examples/02_basic_binary_trait.R
#
# Binary-trait BayesR + BayesA with Albert-Chib data augmentation (probit
# link), using the multi-allelic microhaplotype (MH) marker representation.
# Binary metrics are reported on the OBSERVED (probability) scale:
#   bias  = calibration slope (1.0 = well-calibrated)
#   RMSE  = Brier-like
#   AUC   = rank-invariant, link-independent
#
# Uses the bundled demo dataset (load_data()): 10 full-sib families x 20
# offspring, within-family 80/20 split, QTL effects at the allele level
# (QTL@MH architecture). For SNP / SNP-vs-MH comparison see
# examples/03_marker_QTL_congruency_theory.R.
#
# NEW fields exposed:
#   fit$prob_train   -- training-set predicted probabilities pnorm(GEBV)
#   pred$prob        -- test-set predicted probabilities pnorm(GEBV)
#   pred$metrics$bias = calibration slope on observed scale
#
# Requirements: masbayes, pROC
# Usage: source("examples/02_basic_binary_trait.R")

library(masbayes)

# ── Load bundled demo data and build MH design matrix W_mh ──────────────────
d <- load_data()
bid      <- attr(d$mh, "block_id")
wah_full <- construct_wah_matrix(d$mh, bid, d$allele_freq, NULL, TRUE)
W        <- wah_full$W_ah
n <- nrow(W)

# y_bin is stored as integer in the demo Rds -- coerce to double because the
# Rust MCMC kernel expects numeric (not integer) input.
y_bin <- as.numeric(d$pheno$y_bin_qtl_mh)
X     <- model.matrix(~ sex - 1, data = d$pheno)

cat(sprintf("Prevalence: %.3f | n_cases=%d | n_controls=%d\n",
            mean(y_bin), sum(y_bin), sum(1 - y_bin)))

wtw <- colSums(W ^ 2)

mcmc_p <- list(n_iter = 2000L, n_burn = 1000L, n_thin = 5L, seed = 123L)

# ── BayesR (binary) ──────────────────────────────────────────────────────────
fit_r <- run_bayesr(
  w             = W,
  X             = X,
  y             = y_bin,
  wtw_diag      = wtw,
  pi_vec        = c(0.90, 0.05, 0.03, 0.02),
  sigma2_e_init = 1.0,                        # liability scale
  sigma2_ah     = 1.0,                        # liability scale
  prior_params  = list(a0_e = 10, a0_g = 10,
                       variance_class = c(0, 0.01, 0.1, 1)),
  mcmc_params   = mcmc_p,
  method        = "mcmc",
  response_type = "binary",                   # Albert-Chib augmentation
  save_rds      = FALSE
)

# ── BayesA (binary) ──────────────────────────────────────────────────────────
fit_a <- run_bayesa(
  w             = W,
  X             = X,
  y             = y_bin,
  wtw_diag      = wtw,
  nu            = 4.5,
  sigma2_g      = 1.0,
  sigma2_e_init = 1.0,
  prior_params  = list(a0_e = 10),
  mcmc_params   = mcmc_p,
  method        = "mcmc",
  response_type = "binary",
  save_rds      = FALSE
)

# ── New field: fit$prob_train (binary only) ────────────────────────────────
cat(sprintf("\nfit_r$prob_train range: [%.3f, %.3f]  median=%.3f\n",
            min(fit_r$prob_train), max(fit_r$prob_train),
            median(fit_r$prob_train)))

# ── Full reports (Training fit block now reports observed-scale metrics) ──
cat("\n--- BayesR Summary (binary) ---\n")
summary(fit_r)
cat("\n--- BayesA Summary (binary) ---\n")
summary(fit_a)

# ── In-sample evaluation: predict() reports observed-scale metrics ────────
in_r <- predict(fit_r)
in_a <- predict(fit_a)
cat(sprintf("\n-- In-sample metrics (observed/probability scale) --\n"))
cat(sprintf("  BayesR: AUC=%.3f | RMSE=%.3f | bias=%.3f (calib slope)\n",
            in_r$metrics$AUC, in_r$metrics$RMSE, in_r$metrics$bias))
cat(sprintf("  BayesA: AUC=%.3f | RMSE=%.3f | bias=%.3f (calib slope)\n",
            in_a$metrics$AUC, in_a$metrics$RMSE, in_a$metrics$bias))
cat("  bias = 1.0 ideal; <1 over-dispersion; >1 under-dispersion.\n")

# ── Liability-scale GEBVs and z_hat ────────────────────────────────────────
cat(sprintf("\nz_hat range (BayesR): [%.3f, %.3f]\n",
            min(fit_r$z_hat), max(fit_r$z_hat)))
cat(sprintf("sigma2_e mean (should be ~1.0): %.4f\n",
            mean(fit_r$sigma2_e_samples)))

# ── Train/test split + AUC + calibration on hold-out ──────────────────────
# Bundled within-family split: 16 train + 4 test per family. Rebuild W_mh
# on the train set with d$allele_freq, then reuse the training reference
# structure for the test set so columns stay aligned.
idx_train <- d$train_idx
idx_test  <- d$test_idx

wah_train  <- construct_wah_matrix(d$mh[idx_train, ], bid,
                                   d$allele_freq, NULL, TRUE)
W_train    <- wah_train$W_ah
ref_train  <- list(allele_info     = wah_train$allele_info,
                   dropped_alleles = wah_train$dropped_alleles)
W_test     <- construct_wah_matrix(d$mh[idx_test, ], bid,
                                   NULL, ref_train, TRUE)$W_ah
X_train    <- X[idx_train, , drop = FALSE]
X_test     <- X[idx_test, , drop = FALSE]
y_train    <- y_bin[idx_train]
y_test     <- y_bin[idx_test]

fit_train <- run_bayesr(
  w             = W_train,
  X             = X_train,
  y             = y_train,
  wtw_diag      = colSums(W_train ^ 2),
  pi_vec        = c(0.90, 0.05, 0.03, 0.02),
  sigma2_e_init = 1.0,
  sigma2_ah     = 1.0,
  mcmc_params   = mcmc_p,
  method        = "mcmc",
  response_type = "binary",
  save_rds      = FALSE,
  verbose       = FALSE
)
pred_test <- predict(fit_train, W_test, y_test, X_new = X_test)

cat(sprintf("\n-- Hold-out (BayesR test set, observed/probability scale) --\n"))
cat(sprintf("  AUC          : %.3f\n", pred_test$metrics$AUC))
cat(sprintf("  RMSE (~Brier): %.3f\n", pred_test$metrics$RMSE))
cat(sprintf("  bias (calib) : %.3f\n", pred_test$metrics$bias))
cat(sprintf("  prob range   : [%.3f, %.3f]\n",
            min(pred_test$prob), max(pred_test$prob)))

# ════════════════════════════════════════════════════════════════════════════
# Visualisation -- three panels (ROC + GEBV by class + calibration plot)
# ════════════════════════════════════════════════════════════════════════════

gebv_r <- fit_r$GEBV
gebv_a <- fit_a$GEBV

# Helper: decile-binned calibration curve. n_bins=5 chosen because n=200
# and the demo class balance produces sparse decile bins at n_bins=10.
.calibration_curve <- function(y, prob, n_bins = 5) {
  qs    <- quantile(prob, probs = seq(0, 1, length.out = n_bins + 1),
                    na.rm = TRUE)
  qs[1] <- qs[1] - 1e-9
  bin   <- cut(prob, breaks = qs, include.lowest = TRUE, labels = FALSE)
  agg   <- data.frame(bin = bin, y = y, p = prob)
  aggregate(cbind(y = y, p = p) ~ bin, data = agg, FUN = mean)
}

par(mfrow = c(1, 3))

# Panel 1 -- ROC (rank-based, identical whether scored on liability or prob)
pROC::plot.roc(pROC::roc(y_bin, gebv_r, quiet = TRUE),
               main = sprintf("BayesR ROC (AUC=%.3f)", in_r$metrics$AUC))
pROC::plot.roc(pROC::roc(y_bin, gebv_a, quiet = TRUE),
               main = sprintf("BayesA ROC (AUC=%.3f)", in_a$metrics$AUC),
               add  = FALSE)

# Panel 2 -- Liability GEBV by class
boxplot(gebv_r ~ y_bin,
        names = c("Control (0)", "Case (1)"),
        main  = sprintf("BayesR GEBV (liability)\nbias=%.3f",
                        in_r$metrics$bias),
        ylab  = "Liability scale GEBV",
        col   = c("lightblue", "lightcoral"))

# Panel 3 -- Calibration plot: probability vs observed proportion
cal_r <- .calibration_curve(y_bin, fit_r$prob_train)
cal_a <- .calibration_curve(y_bin, fit_a$prob_train)
plot(cal_r$p, cal_r$y, type = "b", lwd = 2, pch = 19, col = "steelblue",
     xlim = c(0, 1), ylim = c(0, 1),
     main = "Calibration Plot (in-sample)",
     xlab = "Predicted P(y = 1)", ylab = "Observed proportion")
lines(cal_a$p, cal_a$y, type = "b", lwd = 2, pch = 17, col = "firebrick")
abline(0, 1, lty = 2, col = "grey60")
legend("topleft",
       legend = c(
         sprintf("BayesR (slope=%.3f)", in_r$metrics$bias),
         sprintf("BayesA (slope=%.3f)", in_a$metrics$bias)
       ),
       col = c("steelblue", "firebrick"), lwd = 2, pch = c(19, 17),
       bty = "n", cex = 0.85)

par(mfrow = c(1, 1))

cat(sprintf("\nW_mh dimensions (full): %d x %d (allele-level)\n",
            nrow(W), ncol(W)))
cat("\nDone.\n")
