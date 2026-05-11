# examples/01_basic_continuous_trait.R
#
# BayesR + BayesA for a continuous trait, using the multi-allelic
# microhaplotype (MH) marker representation. The bundled demo dataset
# (load_data()) is a 10 full-sib family x 20 offspring population with a
# within-family 80/20 train/test split and QTL effects placed at the
# allele-level W_mh columns -- so this example is the natural pair to
# "QTL@MH architecture".
#
# For the analogous SNP path and a direct SNP-vs-MH comparison study, see
# examples/03_marker_QTL_congruency_theory.R.
#
# Requirements: masbayes
# Usage: source("examples/01_basic_continuous_trait.R")

library(masbayes)

# ── Load bundled demo data ──────────────────────────────────────────────────
d <- load_data()

# Build the MH design matrix W_mh (Da 2015 allele-level encoding) on the FULL
# population. d$allele_freq is the pre-computed training-style frequency
# table required when no reference_structure is supplied.
bid       <- attr(d$mh, "block_id")
wah_full  <- construct_wah_matrix(d$mh, bid, d$allele_freq, NULL, TRUE)
W         <- wah_full$W_ah
ref_full  <- list(allele_info     = wah_full$allele_info,
                  dropped_alleles = wah_full$dropped_alleles)
n <- nrow(W)

# Continuous phenotype matching the marker representation: QTL@MH.
# Fixed effect: sex (factor F/M), passed as a 2-column contrast without
# intercept (run_bayesr samples mu separately).
y <- d$pheno$y_cont_qtl_mh
X <- model.matrix(~ sex - 1, data = d$pheno)

# Sufficient statistic
wtw <- colSums(W ^ 2)

mcmc_p <- list(n_iter = 2000L, n_burn = 1000L, n_thin = 5L, seed = 123L)

# ── BayesR ───────────────────────────────────────────────────────────────────
fit_r <- run_bayesr(
  w             = W,
  X             = X,
  y             = y,
  wtw_diag      = wtw,
  pi_vec        = c(0.90, 0.05, 0.03, 0.02),
  sigma2_e_init = var(y) * 0.5,
  sigma2_ah     = var(y) * 0.5,
  prior_params  = list(a0_e = 10, a0_g = 10,
                       variance_class = c(0, 0.01, 0.1, 1)),
  mcmc_params   = mcmc_p,
  method        = "mcmc",
  save_rds      = FALSE
)

# ── BayesA ───────────────────────────────────────────────────────────────────
fit_a <- run_bayesa(
  w             = W,
  X             = X,
  y             = y,
  wtw_diag      = wtw,
  nu            = 4.5,
  sigma2_g      = var(y) * 0.5,
  sigma2_e_init = var(y) * 0.5,
  prior_params  = list(a0_e = 10),
  mcmc_params   = mcmc_p,
  method        = "mcmc",
  save_rds      = FALSE
)

# ── Full reports via summary() ─────────────────────────────────────────────
cat("\n--- BayesR Summary ---\n")
summary(fit_r)
cat("\n--- BayesA Summary ---\n")
summary(fit_a)

# ── In-sample evaluation via predict() ──────────────────────────────────────
in_r <- predict(fit_r)
in_a <- predict(fit_a)
cat(sprintf("\nIn-sample: BayesR R2=%.3f | BayesA R2=%.3f\n",
            in_r$metrics$R2, in_a$metrics$R2))

# ── Train/test split via predict(newdata, y_new) ───────────────────────────
# Split is bundled in the demo data (within-family 16/4 per family). The MH
# training matrix is rebuilt with d$allele_freq, then the test matrix reuses
# the training reference_structure so the columns stay aligned.
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
y_train    <- y[idx_train]
y_test     <- y[idx_test]

fit_train <- run_bayesr(
  w             = W_train,
  X             = X_train,
  y             = y_train,
  wtw_diag      = colSums(W_train ^ 2),
  pi_vec        = c(0.90, 0.05, 0.03, 0.02),
  sigma2_e_init = var(y_train) * 0.5,
  sigma2_ah     = var(y_train) * 0.5,
  mcmc_params   = mcmc_p,
  method        = "mcmc",
  save_rds      = FALSE,
  verbose       = FALSE
)
pred_test <- predict(fit_train, W_test, y_test, X_new = X_test)

# Standard genomic-prediction accuracy: cor(GEBV, TBV). cor(GEBV, y) is
# bounded by sqrt(h2) so we report both -- breeders use cor(GEBV, TBV)
# because it is not attenuated by residual environmental variance.
tbv_test  <- d$pheno$tbv_qtl_mh[idx_test]
r_test_g  <- cor(pred_test$GEBV, tbv_test)
cat(sprintf(
  "Hold-out (BayesR): r_test_g=%.3f (vs TBV) | accuracy=%.3f (vs y) | bias=%.3f\n",
  r_test_g, pred_test$metrics$accuracy, pred_test$metrics$bias
))

# ── Posterior mixture proportions (BayesR) ─────────────────────────────────
pi_post <- colMeans(fit_r$pi_samples)
names(pi_post) <- c("Zero", "Small", "Medium", "Large")
cat("\nBayesR posterior mixture proportions:\n")
print(round(pi_post, 3))

# ── Convergence diagnostics traces ─────────────────────────────────────────
par(mfrow = c(1, 2))
plot(fit_r$sigma2_e_samples, type = "l",
     main = "BayesR: Residual Variance Trace",
     xlab = "Iteration", ylab = "sigma2_e")
plot(fit_a$sigma2_e_samples, type = "l",
     main = "BayesA: Residual Variance Trace",
     xlab = "Iteration", ylab = "sigma2_e")
par(mfrow = c(1, 1))

cat(sprintf("\nW_mh dimensions (full): %d x %d (allele-level)\n",
            nrow(W), ncol(W)))
