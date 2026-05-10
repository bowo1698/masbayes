# examples/02_basic_binary_trait.R
#
# Binary trait example: BayesR and BayesA with Albert-Chib data augmentation
# (probit link). Demonstrates response_type = "binary" plus the predict()
# / summary() API. Binary metrics are reported on the OBSERVED (probability)
# scale: bias is the calibration slope (1.0 = well-calibrated), RMSE is
# Brier-like, AUC is rank-invariant.
#
# NEW fields exposed:
#   fit$prob_train   — training-set predicted probabilities pnorm(GEBV)
#   pred$prob        — test-set predicted probabilities pnorm(GEBV)
#   pred$metrics$bias = calibration slope on observed scale
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

# ── New field: fit$prob_train (binary only) ────────────────────────────────
# Probability predictions for training set, derived from the liability GEBV
# via probit inverse link. Useful for plotting / threshold tuning without
# calling predict(fit).
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

# pred_te$prob is the NEW probability field; pred_te$GEBV is liability
cat(sprintf("\n-- Hold-out (BayesR test set, observed/probability scale) --\n"))
cat(sprintf("  AUC          : %.3f\n", pred_te$metrics$AUC))
cat(sprintf("  RMSE (~Brier): %.3f\n", pred_te$metrics$RMSE))
cat(sprintf("  bias (calib) : %.3f\n", pred_te$metrics$bias))
cat(sprintf("  prob range   : [%.3f, %.3f]\n",
            min(pred_te$prob), max(pred_te$prob)))

# ════════════════════════════════════════════════════════════════════════════
# Visualisation — three panels (ROC + GEBV by class + calibration plot)
# ════════════════════════════════════════════════════════════════════════════

gebv_r <- fit_r$GEBV
gebv_a <- fit_a$GEBV

# Helper: decile-binned calibration curve
.calibration_curve <- function(y, prob, n_bins = 10) {
  qs    <- quantile(prob, probs = seq(0, 1, length.out = n_bins + 1),
                    na.rm = TRUE)
  qs[1] <- qs[1] - 1e-9
  bin   <- cut(prob, breaks = qs, include.lowest = TRUE, labels = FALSE)
  agg   <- data.frame(bin = bin, y = y, p = prob)
  aggregate(cbind(y = y, p = p) ~ bin, data = agg, FUN = mean)
}

par(mfrow = c(1, 3))

# Panel 1 — ROC (rank-based, identical whether scored on liability or prob)
pROC::plot.roc(pROC::roc(y_bin, gebv_r, quiet = TRUE),
               main = sprintf("BayesR ROC (AUC=%.3f)", in_r$metrics$AUC))
pROC::plot.roc(pROC::roc(y_bin, gebv_a, quiet = TRUE),
               main = sprintf("BayesA ROC (AUC=%.3f)", in_a$metrics$AUC),
               add  = FALSE)

# Panel 2 — Liability GEBV by class
boxplot(gebv_r ~ y_bin,
        names = c("Control (0)", "Case (1)"),
        main  = sprintf("BayesR GEBV (liability)\nbias=%.3f",
                        in_r$metrics$bias),
        ylab  = "Liability scale GEBV",
        col   = c("lightblue", "lightcoral"))

# Panel 3 — Calibration plot (NEW): probability vs observed proportion
# A perfectly calibrated model lies on the diagonal y = x; the slope of
# this curve corresponds to pred_te$metrics$bias / in_*$metrics$bias.
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

cat("\nDone.\n")
