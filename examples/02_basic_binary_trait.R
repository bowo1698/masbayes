# examples/02_basic_binary_trait.R
#
# Binary trait example: BayesR and BayesA with Albert-Chib data augmentation.
# Demonstrates response_type = "binary" plus the new summary() / predict() API.
# AUC is reported automatically by predict() for binary fits.
#
# Requirements: masbayes, pROC
# Usage: source("examples/02_basic_binary_trait.R")

library(masbayes)

set.seed(42)
n <- 200
p <- 100

# Simulate genotype matrix (or substitute construct_snp_matrix(X)$W for SNP dosage)
W <- matrix(rnorm(n * p), n, p)

# Simulate continuous liability and threshold to binary
liability <- W[, 1:5] %*% rnorm(5, 0, 0.5) + rnorm(n, 0, 1)
y_bin     <- as.numeric(liability > median(liability))   # prevalence ~50%

cat(sprintf("Prevalence: %.3f | n_cases=%d | n_controls=%d\n",
            mean(y_bin), sum(y_bin), sum(1 - y_bin)))

wtw <- colSums(W^2)

mcmc_p <- list(n_iter = 2000L, n_burn = 1000L, n_thin = 5L, seed = 123L)

# ── BayesR (binary) ──────────────────────────────────────────────────────────
fit_r <- run_bayesr(
  w             = W,
  y             = y_bin,
  wtw_diag      = wtw,
  pi_vec        = c(0.90, 0.05, 0.03, 0.02),
  sigma2_e_init = 1.0,                        # liability scale
  sigma2_ah     = 1.0,                        # liability scale
  prior_params  = list(a0_e=10, a0_g=10, variance_class=c(0, 0.01, 0.1, 1)),
  mcmc_params   = mcmc_p,
  method        = "mcmc",
  response_type = "binary",                   # Albert-Chib data augmentation
  save_rds      = FALSE
)

# ── BayesA (binary) ──────────────────────────────────────────────────────────
fit_a <- run_bayesa(
  w             = W,
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

# ── Full reports (AUC included automatically) ──────────────────────────────
cat("\n--- BayesR Summary (binary) ---\n")
summary(fit_r)
cat("\n--- BayesA Summary (binary) ---\n")
summary(fit_a)

# ── predict() reports AUC in metrics$AUC ───────────────────────────────────
in_r <- predict(fit_r)
in_a <- predict(fit_a)
cat(sprintf("\nIn-sample AUC: BayesR=%.3f | BayesA=%.3f\n",
            in_r$metrics$AUC, in_a$metrics$AUC))

# ── Liability-scale GEBVs and z_hat ────────────────────────────────────────
cat(sprintf("z_hat range (BayesR): [%.3f, %.3f]\n",
            min(fit_r$z_hat), max(fit_r$z_hat)))
cat(sprintf("sigma2_e mean (should be ~1.0): %.4f\n",
            mean(fit_r$sigma2_e_samples)))

# ── Train/test split + AUC on hold-out ─────────────────────────────────────
idx_tr <- sample(n, 0.8 * n)
fit_tr <- run_bayesr(
  w             = W[idx_tr, ],
  y             = y_bin[idx_tr],
  wtw_diag      = colSums(W[idx_tr, ]^2),
  pi_vec        = c(0.90, 0.05, 0.03, 0.02),
  sigma2_e_init = 1.0,
  sigma2_ah     = 1.0,
  mcmc_params   = mcmc_p,
  method        = "mcmc",
  response_type = "binary",
  save_rds      = FALSE,
  verbose       = FALSE
)
pred_te <- predict(fit_tr, W[-idx_tr, ], y_bin[-idx_tr])
cat(sprintf("Hold-out (BayesR): AUC=%.3f | RMSE=%.3f\n",
            pred_te$metrics$AUC, pred_te$metrics$RMSE))

# ── ROC curves (using GEBV from fit object) ────────────────────────────────
gebv_r <- fit_r$GEBV
gebv_a <- fit_a$GEBV

par(mfrow = c(1, 2))
pROC::plot.roc(pROC::roc(y_bin, gebv_r, quiet = TRUE),
               main = sprintf("BayesR ROC (AUC=%.3f)", in_r$metrics$AUC))
pROC::plot.roc(pROC::roc(y_bin, gebv_a, quiet = TRUE),
               main = sprintf("BayesA ROC (AUC=%.3f)", in_a$metrics$AUC))
par(mfrow = c(1, 1))

# ── Liability distribution by class ────────────────────────────────────────
par(mfrow = c(1, 2))
boxplot(gebv_r ~ y_bin,
        names = c("Control (0)", "Case (1)"),
        main  = "BayesR GEBV by Class",
        ylab  = "Liability scale GEBV")
boxplot(gebv_a ~ y_bin,
        names = c("Control (0)", "Case (1)"),
        main  = "BayesA GEBV by Class",
        ylab  = "Liability scale GEBV")
par(mfrow = c(1, 1))
