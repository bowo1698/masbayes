# R/predict-methods.R
# S3 predict methods for masbayes fit objects.
# Scheme-agnostic: full-fit, train/test split, and CV folds all use the same
# function. Behaviour depends on whether `newdata` and `y_new` are supplied.

#' Predict from a BayesR Fit
#'
#' Computes genomic estimated breeding values (GEBVs) and prediction
#' metrics. Works identically for full-fit, train/test split, and k-fold
#' CV — the caller is responsible for data partitioning; \code{predict()}
#' only evaluates the trained model on whatever data it receives.
#'
#' @details
#' \strong{GEBV formula (direct marker effects).}
#' The genomic estimated breeding value for individual \eqn{i} is obtained
#' by summing the posterior mean allele effects across all loci:
#' \deqn{\hat{g}_i = \hat{\mu} + \mathbf{w}_i^\top \hat{\boldsymbol{\beta}}}
#' where \eqn{\mathbf{w}_i} is row \eqn{i} of the marker design matrix \eqn{W}
#' and \eqn{\hat{\boldsymbol{\beta}}} is the vector of posterior mean allele
#' effects. When fixed effects were supplied to the fitting function:
#' \deqn{\hat{g}_i = \hat{\mu} + \mathbf{x}_i^\top \hat{\boldsymbol{\alpha}} + \mathbf{w}_i^\top \hat{\boldsymbol{\beta}}}
#' For binary traits, the liability-scale GEBV is converted to a predicted
#' probability via the probit inverse link:
#' \deqn{P(y_i = 1) = \Phi(\hat{g}_i)}
#' This is identical for SNP and multi-allelic (microhaplotype) markers;
#' only the construction of \eqn{W} differs between the two marker types.
#'
#' \strong{Three usage modes.}
#' \describe{
#'   \item{\strong{In-sample (full-fit)}}{Call \code{predict(fit)}. Returns
#'     training GEBVs and the in-sample metrics already stored in the fit
#'     object.}
#'   \item{\strong{Forecast only}}{Call \code{predict(fit, W_new)} without
#'     \code{y_new}. Returns GEBVs for the new design matrix; metrics are
#'     skipped.}
#'   \item{\strong{Test evaluation}}{Call \code{predict(fit, W_test,
#'     y_test)}. Returns test-set GEBVs plus \code{R2, RMSE, accuracy/AUC,
#'     bias}.}
#' }
#'
#' For test sets built with \code{\link{construct_wah_matrix}}, pass the
#' training matrix structure as \code{reference_structure} so that
#' \code{newdata} has identical columns to the training matrix.
#'
#' @param object An object of class \code{masbayes_bayesr}.
#' @param newdata Optional numeric design matrix. If \code{NULL}, the
#'   training GEBVs (\code{object$pred_train}) are returned.
#' @param y_new Optional response vector for \code{newdata}. When supplied,
#'   prediction metrics are computed.
#' @param ... Unused.
#'
#' @return An object of class \code{masbayes_prediction}: a list with
#'   \code{GEBV} (liability scale for binary), \code{prob} (binary only:
#'   \code{P(y = 1) = pnorm(GEBV)}, otherwise \code{NULL}),
#'   \code{metrics} (\code{R2, RMSE, accuracy/AUC, bias} or
#'   \code{NULL}; for binary, computed on the observed/probability
#'   scale so \code{bias} is the calibration slope), \code{h2, sigma2_g,
#'   sigma2_e}, \code{variance_components}, \code{response_type},
#'   \code{model_type}, \code{eval_scope}, and \code{has_truth}.
#'
#' @examples
#' \dontrun{
#' d       <- load_data("small")
#' W       <- construct_snp_matrix(d$snp)$W
#' y       <- d$pheno$y_cont_qtl_snp
#' W_train <- W[d$train_idx, ]
#' W_test  <- W[d$test_idx, ]
#' y_train <- y[d$train_idx]
#' y_test  <- y[d$test_idx]
#'
#' fit  <- run_bayesr(
#'   w             = W_train,
#'   y             = y_train,
#'   wtw_diag      = colSums(W_train^2),
#'   sigma2_e_init = var(y_train) / 2,
#'   sigma2_ah     = var(y_train) / 2,
#'   mcmc_params   = list(n_iter = 1000L, n_burn = 500L,
#'                        n_thin = 5L, seed = 1L),
#'   save_rds      = FALSE
#' )
#'
#' pred <- predict(fit, W_test, y_test)   # test-set evaluation
#' pred$metrics$accuracy
#'
#' fc   <- predict(fit, W_test)           # forecast only, no y_new
#' head(fc$GEBV)
#'
#' insamp <- predict(fit)                 # in-sample reference
#' insamp$metrics$R2
#' }
#'
#' @seealso \code{\link{run_bayesr}},
#'   \code{\link{summary.masbayes_bayesr}}
#' @export
predict.masbayes_bayesr <- function(object, newdata = NULL, y_new = NULL,
                                    X_new = NULL, ...) {
  predict_masbayes_common(object, newdata = newdata, y_new = y_new,
                          X_new = X_new)
}

#' Predict from a BayesA Fit
#'
#' Computes GEBVs and prediction metrics from a \code{\link{run_bayesa}}
#' fit. Behaviour and return value are identical to
#' \code{\link{predict.masbayes_bayesr}}; see that page for usage modes
#' and examples.
#'
#' @param object An object of class \code{masbayes_bayesa}.
#' @param newdata Optional numeric design matrix.
#' @param y_new Optional response vector for \code{newdata}.
#' @param ... Unused.
#'
#' @return An object of class \code{masbayes_prediction}.
#'
#' @examples
#' \dontrun{
#' d       <- load_data("small")
#' W       <- construct_snp_matrix(d$snp)$W
#' y_bin   <- as.numeric(d$pheno$y_bin_qtl_snp)
#' W_train <- W[d$train_idx, ]
#' W_test  <- W[d$test_idx, ]
#'
#' fit  <- run_bayesa(
#'   w             = W_train,
#'   y             = y_bin[d$train_idx],
#'   wtw_diag      = colSums(W_train^2),
#'   nu            = 4.5,
#'   sigma2_g      = 1.0,
#'   sigma2_e_init = 1.0,
#'   mcmc_params   = list(n_iter = 1000L, n_burn = 500L,
#'                        n_thin = 5L, seed = 1L),
#'   response_type = "binary",
#'   save_rds      = FALSE
#' )
#' pred <- predict(fit, W_test, y_bin[d$test_idx])
#' pred$metrics$AUC
#' }
#'
#' @seealso \code{\link{run_bayesa}},
#'   \code{\link{summary.masbayes_bayesa}},
#'   \code{\link{predict.masbayes_bayesr}}
#' @export
predict.masbayes_bayesa <- function(object, newdata = NULL, y_new = NULL,
                                    X_new = NULL, ...) {
  predict_masbayes_common(object, newdata = newdata, y_new = y_new,
                          X_new = X_new)
}

#' @keywords internal
#' @noRd
predict_masbayes_common <- function(object, newdata = NULL, y_new = NULL,
                                    X_new = NULL) {
  beta      <- as.numeric(object$beta_hat)
  mu        <- as.numeric(object$mu_hat %||% 0)
  alpha_hat <- object$alpha_hat   # NULL when fit had no fixed effects
  has_fe    <- !is.null(alpha_hat) && length(alpha_hat) > 0L

  if (is.null(newdata)) {
    GEBV       <- as.numeric(object$pred_train)
    eval_scope <- "in-sample (training)"
    in_sample  <- TRUE
  } else {
    if (!is.matrix(newdata)) newdata <- as.matrix(newdata)
    if (ncol(newdata) != length(beta)) {
      stop(sprintf(
        "ncol(newdata) = %d does not match length(beta_hat) = %d",
        ncol(newdata), length(beta)
      ))
    }
    GEBV <- as.numeric(newdata %*% beta + mu)

    if (has_fe) {
      if (is.null(X_new))
        stop("Fit was trained with fixed effects (X). ",
             "Provide X_new with ncol = length(alpha_hat) for prediction.")
      if (!is.matrix(X_new)) X_new <- as.matrix(X_new)
      if (ncol(X_new) != length(alpha_hat))
        stop(sprintf(
          "ncol(X_new) = %d does not match length(alpha_hat) = %d",
          ncol(X_new), length(alpha_hat)
        ))
      if (nrow(X_new) != nrow(newdata))
        stop(sprintf(
          "nrow(X_new) = %d does not match nrow(newdata) = %d",
          nrow(X_new), nrow(newdata)
        ))
      GEBV <- GEBV + as.numeric(X_new %*% as.numeric(alpha_hat))
    }

    eval_scope <- "test set (newdata)"
    in_sample  <- FALSE
  }

  metrics <- NULL
  if (in_sample) {
    metrics <- object$training_metrics
  } else if (!is.null(y_new)) {
    if (length(y_new) != length(GEBV)) {
      stop(sprintf(
        "length(y_new) = %d does not match nrow(newdata) = %d",
        length(y_new), length(GEBV)
      ))
    }
    # Binary GEBV is on liability scale; pnorm() → observed-scale prob.
    metrics <- if (identical(object$response_type, "binary"))
      compute_metrics_binary(as.numeric(y_new), stats::pnorm(GEBV))
    else
      compute_metrics_gaussian(as.numeric(y_new), GEBV)
  }

  prob <- if (identical(object$response_type, "binary")) {
    stats::pnorm(GEBV)
  } else {
    NULL
  }

  out <- list(
    GEBV                = GEBV,
    prob                = prob,
    metrics             = metrics,
    h2                  = object$h2,
    sigma2_g            = object$sigma2_g,
    sigma2_e            = object$sigma2_e,
    variance_components = object$variance_components,
    response_type       = object$response_type,
    model_type          = object$model_type,
    eval_scope          = eval_scope,
    has_truth           = !is.null(metrics)
  )
  class(out) <- "masbayes_prediction"
  out
}

#' Print masbayes_prediction objects (S3 dispatch).
#' @noRd
#' @export
print.masbayes_prediction <- function(x, ...) {
  model_label <- switch(x$model_type,
                        bayesr = "BayesR", bayesa = "BayesA", "masbayes")
  cat("\n--- masbayes prediction ---\n")
  cat(sprintf(" Model         : %s\n", model_label))
  cat(sprintf(" Response type : %s\n", x$response_type))
  cat(sprintf(" Evaluation    : %s\n", x$eval_scope))
  cat(sprintf(" GEBV (n=%d)   : min=%s  median=%s  max=%s\n",
              length(x$GEBV),
              fmt_num(min(x$GEBV)),
              fmt_num(stats::median(x$GEBV)),
              fmt_num(max(x$GEBV))))

  if (identical(x$response_type, "binary") && !is.null(x$prob)) {
    cat(sprintf(" Prob (P(y=1)) : min=%s  median=%s  max=%s\n",
                fmt_num(min(x$prob)),
                fmt_num(stats::median(x$prob)),
                fmt_num(max(x$prob))))
  }

  if (!is.null(x$metrics)) {
    if (identical(x$response_type, "binary")) {
      cat(" Metrics (observed/probability scale)\n")
      cat(sprintf("   AUC          : %s\n", fmt_num(x$metrics$AUC)))
    } else {
      cat(" Metrics\n")
      cat(sprintf("   accuracy (r) : %s\n", fmt_num(x$metrics$accuracy)))
    }
    cat(sprintf("   R^2          : %s\n", fmt_num(x$metrics$R2)))
    cat(sprintf("   RMSE         : %s\n", fmt_num(x$metrics$RMSE)))
    cat(sprintf("   bias         : %s\n", fmt_num(x$metrics$bias)))
  } else {
    cat(" (no y_new supplied — metrics not computed)\n")
  }

  cat(sprintf(" h2 (from fit)  : %s\n", fmt_num(x$h2)))
  cat(sprintf(" sigma2_g       : %s\n", fmt_num(x$sigma2_g)))
  cat(sprintf(" sigma2_e       : %s\n", fmt_num(x$sigma2_e)))
  cat("---\n\n")
  invisible(x)
}
