# examples/01_basic_continuous_trait.R
#
# Basic example: BayesR and BayesA for continuous trait
# Uses simulated data — no real genotype required
#
# Requirements: masbayes
# Usage: source("examples/01_basic_continuous_trait.R")

library(masbayes)

set.seed(42)
n <- 200
p <- 100

# Simulate genotype matrix and phenotype
W <- matrix(rnorm(n * p), n, p)
y <- W[, 1:5] %*% rnorm(5, 0, 0.5) + rnorm(n, 0, 1)

# Precompute sufficient statistics
wtw <- colSums(W^2)
wty <- as.vector(crossprod(W, y))

# ── BayesR ───────────────────────────────────────────────────────────────────
res_r <- run_bayesr(
  w             = W,
  y             = y,
  wtw_diag      = wtw,
  wty           = wty,
  pi_vec        = c(0.90, 0.05, 0.03, 0.02),
  sigma2_e_init = var(y) * 0.5,
  sigma2_ah     = var(y) * 0.5,
  prior_params  = list(a0_e=10, a0_g=10, variance_class=c(0, 0.01, 0.1, 1)),
  mcmc_params   = list(n_iter=2000L, n_burn=1000L, n_thin=5L, seed=123L),
  method        = "mcmc"
)

# ── BayesA ───────────────────────────────────────────────────────────────────
res_a <- run_bayesa(
  w             = W,
  y             = y,
  wtw_diag      = wtw,
  wty           = wty,
  nu            = 4.5,
  sigma2_g      = var(y) * 0.5,
  sigma2_e_init = var(y) * 0.5,
  prior_params  = list(a0_e=10),
  mcmc_params   = list(n_iter=2000L, n_burn=1000L, n_thin=5L, seed=123L),
  method        = "mcmc"
)

# ── Results ───────────────────────────────────────────────────────────────────
gebv_r <- as.vector(W %*% res_r$beta_hat) + res_r$mu_hat
gebv_a <- as.vector(W %*% res_a$beta_hat) + res_a$mu_hat

cat("=== Continuous Trait Results ===\n")
cat(sprintf("BayesR | r_train=%.3f | h2=%.3f\n", cor(gebv_r, y), res_r$h2))
cat(sprintf("BayesA | r_train=%.3f | h2=%.3f\n", cor(gebv_a, y), res_a$h2))

# Posterior mixture proportions (BayesR)
pi_post <- colMeans(res_r$pi_samples)
names(pi_post) <- c("Zero", "Small", "Medium", "Large")
cat("\nBayesR mixture proportions:\n")
print(round(pi_post, 3))

# Convergence diagnostics
par(mfrow=c(1,2))
plot(res_r$sigma2_e_samples, type="l",
     main="BayesR: Residual Variance Trace",
     xlab="Iteration", ylab="sigma2_e")
plot(res_a$sigma2_e_samples, type="l",
     main="BayesA: Residual Variance Trace",
     xlab="Iteration", ylab="sigma2_e")
par(mfrow=c(1,1))