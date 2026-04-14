# examples/02_basic_binary_trait.R
#
# Binary trait example: BayesR and BayesA with Albert-Chib data augmentation
# Demonstrates response_type = "binary" for 0/1 phenotypes
# GEBV and z_hat are on the liability scale
#
# Requirements: masbayes, pROC
# Usage: source("examples/02_basic_binary_trait.R")

library(masbayes)

set.seed(42)
n <- 200
p <- 100

# Simulate genotype matrix
W <- matrix(rnorm(n * p), n, p)

# Simulate continuous liability and threshold to binary
liability <- W[, 1:5] %*% rnorm(5, 0, 0.5) + rnorm(n, 0, 1)
y_bin     <- as.numeric(liability > median(liability))  # prevalence ~50%

cat(sprintf("Prevalence: %.3f | n_cases=%d | n_controls=%d\n",
            mean(y_bin), sum(y_bin), sum(1 - y_bin)))

# Precompute sufficient statistics from binary y
wtw <- colSums(W^2)
wty <- as.vector(crossprod(W, y_bin))

mcmc_p <- list(n_iter=2000L, n_burn=1000L, n_thin=5L, seed=123L)

# ── BayesR (binary) ───────────────────────────────────────────────────────────
res_r <- run_bayesr(
  w             = W,
  y             = y_bin,
  wtw_diag      = wtw,
  wty           = wty,
  pi_vec        = c(0.90, 0.05, 0.03, 0.02),
  sigma2_e_init = 1.0,          # fixed at 1.0 for liability scale
  sigma2_ah     = 1.0,          # prior genetic variance on liability scale
  prior_params  = list(a0_e=10, a0_g=10, variance_class=c(0, 0.01, 0.1, 1)),
  mcmc_params   = mcmc_p,
  method        = "mcmc",
  response_type = "binary"      # Albert-Chib data augmentation
)

# ── BayesA (binary) ───────────────────────────────────────────────────────────
res_a <- run_bayesa(
  w             = W,
  y             = y_bin,
  wtw_diag      = wtw,
  wty           = wty,
  nu            = 4.5,
  sigma2_g      = 1.0,
  sigma2_e_init = 1.0,
  prior_params  = list(a0_e=10),
  mcmc_params   = mcmc_p,
  method        = "mcmc",
  response_type = "binary"
)

# ── Results ───────────────────────────────────────────────────────────────────
gebv_r <- as.vector(W %*% res_r$beta_hat) + res_r$mu_hat
gebv_a <- as.vector(W %*% res_a$beta_hat) + res_a$mu_hat

# AUC
auc_r <- as.numeric(pROC::auc(pROC::roc(y_bin, gebv_r, quiet=TRUE)))
auc_a <- as.numeric(pROC::auc(pROC::roc(y_bin, gebv_a, quiet=TRUE)))

cat("\n=== Binary Trait Results (liability scale) ===\n")
cat(sprintf("BayesR | r_train=%.3f | h2=%.3f | AUC=%.3f\n",
            cor(gebv_r, res_r$z_hat), res_r$h2, auc_r))
cat(sprintf("BayesA | r_train=%.3f | h2=%.3f | AUC=%.3f\n",
            cor(gebv_a, res_a$z_hat), res_a$h2, auc_a))

# z_hat: posterior mean latent liability
cat(sprintf("\nz_hat range (BayesR): [%.3f, %.3f]\n",
            min(res_r$z_hat), max(res_r$z_hat)))

# sigma2_e should be fixed at 1.0 for binary
cat(sprintf("sigma2_e mean (should be 1.0): %.4f\n",
            mean(res_r$sigma2_e_samples)))

# ROC curve
par(mfrow=c(1,2))
pROC::plot.roc(pROC::roc(y_bin, gebv_r, quiet=TRUE),
               main=sprintf("BayesR ROC (AUC=%.3f)", auc_r))
pROC::plot.roc(pROC::roc(y_bin, gebv_a, quiet=TRUE),
               main=sprintf("BayesA ROC (AUC=%.3f)", auc_a))
par(mfrow=c(1,1))

# Liability distribution by class
par(mfrow=c(1,2))
boxplot(gebv_r ~ y_bin,
        names=c("Control (0)", "Case (1)"),
        main="BayesR GEBV by Class",
        ylab="Liability scale GEBV")
boxplot(gebv_a ~ y_bin,
        names=c("Control (0)", "Case (1)"),
        main="BayesA GEBV by Class",
        ylab="Liability scale GEBV")
par(mfrow=c(1,1))