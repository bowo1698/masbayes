# examples/01_basic_continuous_trait.R
#
# Basic example: BayesR and BayesA for a continuous trait.
# Demonstrates the new summary() and predict() S3 methods plus
# construct_snp_matrix() for SNP-style design matrices.
# Uses simulated data — no real genotype required.
#
# Requirements: masbayes
# Usage: source("examples/01_basic_continuous_trait.R")

library(masbayes)

set.seed(42)
n <- 200
p <- 100

# ── Two equivalent ways to build the design matrix ──────────────────────────

# (A) Direct simulation: any numeric matrix is accepted.
W <- matrix(rnorm(n * p), n, p)

# (B) SNP-style: simulate dosage and centre via construct_snp_matrix().
# X <- matrix(rbinom(n * p, 2, prob = runif(p, 0.1, 0.5)), n, p)
# W <- construct_snp_matrix(X)$W

# Phenotype: 5 causal markers + Gaussian noise
y <- W[, 1:5] %*% rnorm(5, 0, 0.5) + rnorm(n, 0, 1)

# Sufficient statistic (precomputed once)
wtw <- colSums(W^2)

mcmc_p <- list(n_iter = 2000L, n_burn = 1000L, n_thin = 5L, seed = 123L)

# ── BayesR ───────────────────────────────────────────────────────────────────
fit_r <- run_bayesr(
  w             = W,
  y             = y,
  wtw_diag      = wtw,
  pi_vec        = c(0.90, 0.05, 0.03, 0.02),
  sigma2_e_init = var(y) * 0.5,
  sigma2_ah     = var(y) * 0.5,
  prior_params  = list(a0_e=10, a0_g=10, variance_class=c(0, 0.01, 0.1, 1)),
  mcmc_params   = mcmc_p,
  method        = "mcmc",
  save_rds      = FALSE
)

# ── BayesA ───────────────────────────────────────────────────────────────────
fit_a <- run_bayesa(
  w             = W,
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
in_r <- predict(fit_r)        # newdata = NULL → uses pred_train
in_a <- predict(fit_a)
cat(sprintf("\nIn-sample: BayesR R2=%.3f | BayesA R2=%.3f\n",
            in_r$metrics$R2, in_a$metrics$R2))

# ── Train/test split via predict(newdata, y_new) ───────────────────────────
idx_tr <- sample(n, 0.8 * n)
fit_tr <- run_bayesr(
  w             = W[idx_tr, ],
  y             = y[idx_tr],
  wtw_diag      = colSums(W[idx_tr, ]^2),
  pi_vec        = c(0.90, 0.05, 0.03, 0.02),
  sigma2_e_init = var(y[idx_tr]) * 0.5,
  sigma2_ah     = var(y[idx_tr]) * 0.5,
  mcmc_params   = mcmc_p,
  method        = "mcmc",
  save_rds      = FALSE,
  verbose       = FALSE
)
pred_te <- predict(fit_tr, W[-idx_tr, ], y[-idx_tr])
cat(sprintf("Hold-out (BayesR): accuracy=%.3f | RMSE=%.3f | bias=%.3f\n",
            pred_te$metrics$accuracy, pred_te$metrics$RMSE,
            pred_te$metrics$bias))

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
