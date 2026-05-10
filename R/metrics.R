# R/metrics.R
# Internal helpers for post-fit metrics, diagnostics, and variance components.
# These are not exported; they are called by run_bayesr(), run_bayesa(),
# summary.masbayes_*(), and predict.masbayes_*().

#' @keywords internal
#' @noRd
`%||%` <- function(a, b) if (is.null(a)) b else a

#' Compute prediction metrics for a Gaussian (continuous) response.
#' @keywords internal
#' @noRd
compute_metrics_gaussian <- function(y, yhat) {
  stopifnot(length(y) == length(yhat))
  if (length(y) < 2L || stats::sd(yhat) < .Machine$double.eps) {
    return(list(
      R2       = NA_real_,
      RMSE     = sqrt(mean((y - yhat)^2)),
      accuracy = NA_real_,
      bias     = NA_real_
    ))
  }
  resid    <- y - yhat
  rmse     <- sqrt(mean(resid^2))
  accuracy <- as.numeric(stats::cor(y, yhat))
  R2       <- accuracy^2
  bias     <- as.numeric(stats::coef(stats::lm(y ~ yhat))[2L])
  list(R2 = R2, RMSE = rmse, accuracy = accuracy, bias = bias)
}

#' Compute prediction metrics for a binary response.
#'
#' \code{yhat} must be on the \strong{observed (probability) scale} —
#' i.e. \code{P(y = 1)}, not the liability-scale linear predictor. Apply
#' \code{pnorm()} (probit, the link used by Albert-Chib augmentation in
#' masbayes) to the liability prediction before calling. Using probabilities
#' makes \code{bias} the calibration slope (1.0 = perfectly calibrated)
#' and \code{RMSE\^2} the Brier score; AUC is rank-invariant under a
#' monotonic inverse-link so it is unchanged.
#' @keywords internal
#' @noRd
compute_metrics_binary <- function(y, yhat) {
  stopifnot(length(y) == length(yhat))
  y_bin <- as.integer(y > 0.5)
  if (length(unique(y_bin)) < 2L) {
    auc <- NA_real_
  } else {
    auc <- tryCatch(
      as.numeric(pROC::auc(pROC::roc(y_bin, as.numeric(yhat),
                                     quiet = TRUE, direction = "<"))),
      error = function(e) NA_real_
    )
  }
  resid    <- y - yhat
  rmse     <- sqrt(mean(resid^2))
  accuracy <- if (stats::sd(yhat) < .Machine$double.eps) NA_real_ else
              as.numeric(stats::cor(y, yhat))
  R2       <- if (is.na(accuracy)) NA_real_ else accuracy^2
  bias     <- if (is.na(accuracy)) NA_real_ else
              as.numeric(stats::coef(stats::lm(y ~ yhat))[2L])
  list(R2 = R2, RMSE = rmse, accuracy = accuracy, AUC = auc, bias = bias)
}

#' ESS and Geweke Z for a numeric MCMC chain.
#' @keywords internal
#' @noRd
chain_diagnostic <- function(x) {
  x <- as.numeric(x)
  if (length(x) < 10L || stats::sd(x) < .Machine$double.eps) {
    return(list(ess = NA_real_, geweke = NA_real_))
  }
  mc  <- coda::as.mcmc(x)
  ess <- tryCatch(as.numeric(coda::effectiveSize(mc)),
                  error = function(e) NA_real_)
  gz  <- tryCatch(as.numeric(coda::geweke.diag(mc)$z),
                  error = function(e) NA_real_)
  list(ess = ess, geweke = gz)
}

#' Diagnostics for a matrix of MCMC samples (rows = draws, cols = parameters).
#' Returns summary stats (min/median/max) of per-column ESS and Geweke.
#' @keywords internal
#' @noRd
matrix_diagnostic <- function(M) {
  if (is.null(M) || !is.matrix(M) || nrow(M) < 10L) {
    return(list(ess_min = NA_real_, ess_median = NA_real_, ess_max = NA_real_,
                geweke_median = NA_real_, geweke_absmax = NA_real_))
  }
  per_col <- apply(M, 2L, chain_diagnostic)
  ess_vec <- vapply(per_col, function(d) d$ess, numeric(1))
  gz_vec  <- vapply(per_col, function(d) d$geweke, numeric(1))
  list(
    ess_min       = suppressWarnings(min(ess_vec, na.rm = TRUE)),
    ess_median    = stats::median(ess_vec, na.rm = TRUE),
    ess_max       = suppressWarnings(max(ess_vec, na.rm = TRUE)),
    geweke_median = stats::median(gz_vec, na.rm = TRUE),
    geweke_absmax = suppressWarnings(max(abs(gz_vec), na.rm = TRUE))
  )
}

#' Compute MCMC diagnostics for a fit list.
#' @keywords internal
#' @noRd
compute_diagnostics <- function(fit, model_type, method) {
  if (method != "mcmc") {
    return(list(method = "em", note = "ESS/Geweke not applicable for EM"))
  }

  out <- list(
    sigma2_e = chain_diagnostic(fit$sigma2_e_samples),
    mu       = chain_diagnostic(fit$mu_samples),
    beta     = matrix_diagnostic(fit$beta_samples)
  )

  if (identical(model_type, "bayesr")) {
    out$sigma2_small  <- chain_diagnostic(fit$sigma2_small_samples)
    out$sigma2_medium <- chain_diagnostic(fit$sigma2_medium_samples)
    out$sigma2_large  <- chain_diagnostic(fit$sigma2_large_samples)
    out$pi            <- matrix_diagnostic(fit$pi_samples)
  } else if (identical(model_type, "bayesa")) {
    out$sigma2_j <- matrix_diagnostic(fit$sigma2_j_samples)
  }

  out
}

#' Variance components for BayesR (4 mixture classes).
#' Reports posterior mean & 95% CI for each class variance and pi proportions.
#' @keywords internal
#' @noRd
varcomp_bayesr <- function(fit, method) {
  cls_summary <- function(samples) {
    if (is.null(samples) || length(samples) == 0L) {
      return(list(mean = NA_real_, lower = NA_real_, upper = NA_real_))
    }
    if (method == "em" || length(samples) < 2L) {
      return(list(mean = mean(samples), lower = NA_real_, upper = NA_real_))
    }
    qs <- stats::quantile(samples, c(0.025, 0.975), names = FALSE, na.rm = TRUE)
    list(mean = mean(samples), lower = qs[1L], upper = qs[2L])
  }

  pi_mean <- if (!is.null(fit$pi_samples)) {
    if (is.matrix(fit$pi_samples)) colMeans(fit$pi_samples) else as.numeric(fit$pi_samples)
  } else rep(NA_real_, 4L)

  list(
    small  = cls_summary(fit$sigma2_small_samples),
    medium = cls_summary(fit$sigma2_medium_samples),
    large  = cls_summary(fit$sigma2_large_samples),
    pi     = list(zero   = pi_mean[1L],
                  small  = pi_mean[2L],
                  medium = pi_mean[3L],
                  large  = pi_mean[4L])
  )
}

#' Variance components for BayesA via tertile binning of sigma2_j_hat.
#' Splits per-marker variances into low/mid/high tertiles.
#' @keywords internal
#' @noRd
varcomp_bayesa_tertile <- function(fit) {
  v <- as.numeric(fit$sigma2_j_hat)
  if (length(v) == 0L) {
    empty <- list(mean = NA_real_, n_markers = 0L,
                  range_lower = NA_real_, range_upper = NA_real_)
    return(list(small = empty, medium = empty, large = empty))
  }
  if (length(v) < 3L) {
    one <- list(mean = mean(v), n_markers = length(v),
                range_lower = min(v), range_upper = max(v))
    return(list(small = one, medium = one, large = one))
  }
  cuts <- stats::quantile(v, c(1/3, 2/3), names = FALSE, na.rm = TRUE)
  bin  <- cut(v, breaks = c(-Inf, cuts[1L], cuts[2L], Inf),
              include.lowest = TRUE, labels = c("small", "medium", "large"))
  bin_summary <- function(label) {
    sel <- v[bin == label]
    if (length(sel) == 0L) {
      return(list(mean = NA_real_, n_markers = 0L,
                  range_lower = NA_real_, range_upper = NA_real_))
    }
    list(mean        = mean(sel),
         n_markers   = length(sel),
         range_lower = min(sel),
         range_upper = max(sel))
  }
  list(small  = bin_summary("small"),
       medium = bin_summary("medium"),
       large  = bin_summary("large"))
}

#' Brief post-fit log printed when verbose = TRUE.
#' Always English. Shows model + algorithm, runtime, h2, and basic diagnostics.
#' @keywords internal
#' @noRd
print_run_summary <- function(fit) {
  model_name <- switch(fit$model_type,
                       bayesr = "BayesR", bayesa = "BayesA", "masbayes")
  algo       <- toupper(fit$method)
  trait_tag  <- if (identical(fit$response_type, "binary")) " (binary trait)" else ""
  rds_line   <- if (!is.null(fit$rds_path))
    sprintf(" RDS saved to    : %s\n", fit$rds_path) else ""

  cat("\n============================================================\n")
  cat(sprintf(" masbayes — %s with %s algorithm%s\n", model_name, algo, trait_tag))
  cat("============================================================\n")
  cat(sprintf(" Response type   : %s\n", fit$response_type))
  cat(sprintf(" Observations    : n = %d, alleles p = %d\n", fit$n, fit$p))
  cat(sprintf(" Runtime         : %.2f seconds\n", fit$runtime))
  cat(rds_line)
  cat(sprintf("\n Heritability (h2): %.3f\n", fit$h2))

  if (identical(fit$method, "mcmc")) {
    d <- fit$diagnostics
    cat("\n MCMC Diagnostics\n")
    cat(" ----------------------------------------\n")
    cat(sprintf("   %-16s %8s   %9s\n", "Parameter", "ESS", "Geweke Z"))
    cat(sprintf("   %-16s %8.1f   %9.2f\n", "sigma2_e",
                d$sigma2_e$ess %||% NA_real_, d$sigma2_e$geweke %||% NA_real_))
    cat(sprintf("   %-16s %8.1f   %9.2f\n", "mu",
                d$mu$ess %||% NA_real_, d$mu$geweke %||% NA_real_))
    cat(sprintf("   %-16s %8.1f   %9.2f\n", "beta (median)",
                d$beta$ess_median %||% NA_real_,
                d$beta$geweke_median %||% NA_real_))
    cat(" ----------------------------------------\n")
  } else {
    cat(sprintf("\n EM Convergence  : completed in fixed iterations (see fit$em_params)\n"))
  }

  cat("\n Run summary(fit) for full report.\n")
  cat("============================================================\n\n")
  invisible(NULL)
}

#' Default RDS filename for a model.
#' @keywords internal
#' @noRd
default_rds_path <- function(model_type) {
  file.path(getwd(), sprintf("results_%s.Rds", model_type))
}

#' Attach post-fit summary fields onto the raw Rust fit list.
#' Used by both run_bayesr() and run_bayesa() wrappers.
#' @keywords internal
#' @noRd
finalise_fit <- function(raw, w, y, model_type, method, response_type,
                         fold_id, runtime, mcmc_params, em_params, call) {
  is_binary <- identical(response_type, "binary")

  yhat_train <- as.numeric(raw$pred_train)
  if (is_binary) {
    # Albert-Chib augmentation uses N(eta, 1) latents → probit link.
    # Convert liability predictions to probabilities before scoring so that
    # bias is the calibration slope and RMSE^2 ≈ Brier score.
    prob_train <- stats::pnorm(yhat_train)
    raw$prob_train <- prob_train
    metrics <- compute_metrics_binary(y, prob_train)
  } else {
    metrics <- compute_metrics_gaussian(y, yhat_train)
  }

  diagnostics <- compute_diagnostics(raw, model_type, method)

  varcomp <- if (identical(model_type, "bayesr"))
               varcomp_bayesr(raw, method)
             else
               varcomp_bayesa_tertile(raw)

  raw$GEBV                <- yhat_train
  raw$runtime             <- as.numeric(runtime)
  raw$training_metrics    <- metrics
  raw$diagnostics         <- diagnostics
  raw$variance_components <- varcomp
  raw$sigma2_e            <- as.numeric(raw$sigma2_e_hat)
  raw$call                <- call
  raw$method              <- method
  raw$response_type       <- response_type
  raw$n                   <- length(y)
  raw$p                   <- ncol(w)
  raw$fold_id             <- fold_id
  raw$model_type          <- model_type
  raw$mcmc_params         <- mcmc_params
  raw$em_params           <- em_params

  cls <- if (identical(model_type, "bayesr")) "masbayes_bayesr" else "masbayes_bayesa"
  class(raw) <- c(cls, "masbayes")
  raw
}

#' Save a fit to RDS and return the resolved path (or NULL if disabled).
#' @keywords internal
#' @noRd
maybe_save_rds <- function(fit, save_rds, save_path) {
  if (!isTRUE(save_rds)) return(NULL)
  path <- save_path %||% default_rds_path(fit$model_type)
  saveRDS(fit, file = path)
  path
}
