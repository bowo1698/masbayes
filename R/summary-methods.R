# R/summary-methods.R
# S3 summary methods for masbayes fit objects.

#' Summarise a BayesR fit (S3 dispatch). See \code{\link{run_bayesr}} for examples.
#' @noRd
#' @export
summary.masbayes_bayesr <- function(object, top_k = 10L, ...) {
  out <- summarise_masbayes_common(object, top_k = top_k)
  out$variance_components <- object$variance_components  # 4-class structure
  class(out) <- c("summary.masbayes_bayesr", "summary.masbayes")
  out
}

#' Summarise a BayesA fit (S3 dispatch). See \code{\link{run_bayesa}} for examples.
#' @noRd
#' @export
summary.masbayes_bayesa <- function(object, top_k = 10L, ...) {
  out <- summarise_masbayes_common(object, top_k = top_k)
  out$variance_components <- object$variance_components  # tertile structure
  class(out) <- c("summary.masbayes_bayesa", "summary.masbayes")
  out
}

#' Internal: shared fields between BayesR and BayesA summaries.
#' @keywords internal
#' @noRd
summarise_masbayes_common <- function(object, top_k = 10L) {
  beta <- as.numeric(object$beta_hat)
  ord  <- order(abs(beta), decreasing = TRUE)
  k    <- min(top_k, length(beta))
  top  <- data.frame(
    rank   = seq_len(k),
    index  = ord[seq_len(k)],
    beta   = beta[ord[seq_len(k)]],
    stringsAsFactors = FALSE
  )

  beta_dim <- if (is.matrix(object$beta_samples)) dim(object$beta_samples) else NULL

  list(
    model_type       = object$model_type,
    method           = object$method,
    response_type    = object$response_type,
    n                = object$n,
    p                = object$p,
    fold_id          = object$fold_id,
    runtime          = object$runtime,
    rds_path         = object$rds_path,

    h2               = object$h2,
    sigma2_g         = object$sigma2_g,
    sigma2_e         = object$sigma2_e,

    training_metrics = object$training_metrics,
    diagnostics      = object$diagnostics,

    beta_hat_summary = list(
      length    = length(beta),
      n_nonzero = sum(beta != 0),
      mean_abs  = mean(abs(beta)),
      max_abs   = max(abs(beta))
    ),
    beta_samples_dim = beta_dim,
    sigma2_j_dim     = if (is.matrix(object$sigma2_j_samples))
                         dim(object$sigma2_j_samples) else NULL,
    GEBV_summary     = stats::quantile(object$GEBV,
                                       probs = c(0, 0.25, 0.5, 0.75, 1),
                                       na.rm = TRUE),
    top_alleles      = top
  )
}

#' Print method for masbayes summaries (S3 dispatch).
#' @noRd
#' @export
print.summary.masbayes <- function(x, ...) {
  model_label <- switch(x$model_type,
                        bayesr = "BayesR", bayesa = "BayesA", "masbayes")
  algo        <- toupper(x$method)
  trait_tag   <- if (identical(x$response_type, "binary")) " (binary trait)" else ""

  cat("\n============================================================\n")
  cat(sprintf(" masbayes Summary — %s with %s%s\n",
              model_label, algo, trait_tag))
  cat("============================================================\n")

  ## Model info ---------------------------------------------------------
  cat(" Model info\n")
  cat(" ----------------------------------------\n")
  cat(sprintf("   Response type : %s\n", x$response_type))
  cat(sprintf("   Observations  : n = %d\n", x$n))
  cat(sprintf("   Alleles       : p = %d\n", x$p))
  cat(sprintf("   Fold id       : %d\n", x$fold_id))
  cat(sprintf("   Runtime       : %.2f seconds\n", x$runtime))
  if (!is.null(x$rds_path))
    cat(sprintf("   RDS path      : %s\n", x$rds_path))

  ## Training fit -------------------------------------------------------
  is_binary <- identical(x$response_type, "binary")
  cat(if (is_binary)
        "\n Training fit (observed/probability scale)\n"
      else
        "\n Training fit\n")
  cat(" ----------------------------------------\n")
  m <- x$training_metrics
  cat(sprintf("   h2            : %.4f\n", x$h2))
  cat(sprintf("   sigma2_g      : %.4f\n", x$sigma2_g))
  cat(sprintf("   sigma2_e      : %.4f\n", x$sigma2_e))
  if (is_binary) {
    cat(sprintf("   AUC           : %s\n", fmt_num(m$AUC)))
  } else {
    cat(sprintf("   accuracy (r)  : %s\n", fmt_num(m$accuracy)))
  }
  cat(sprintf("   R^2           : %s\n", fmt_num(m$R2)))
  cat(sprintf("   RMSE          : %s\n", fmt_num(m$RMSE)))
  cat(sprintf("   bias (slope)  : %s\n", fmt_num(m$bias)))

  ## Variance components ------------------------------------------------
  cat("\n Variance components\n")
  cat(" ----------------------------------------\n")
  if (identical(x$model_type, "bayesr")) {
    print_varcomp_bayesr(x$variance_components)
  } else {
    print_varcomp_bayesa(x$variance_components)
  }

  ## Diagnostics --------------------------------------------------------
  if (identical(x$method, "mcmc")) {
    cat("\n MCMC diagnostics (ESS / Geweke Z)\n")
    cat(" ----------------------------------------\n")
    print_diag_row("sigma2_e",    x$diagnostics$sigma2_e)
    print_diag_row("mu",          x$diagnostics$mu)
    print_diag_matrix_row("beta", x$diagnostics$beta)
    if (identical(x$model_type, "bayesr")) {
      print_diag_row("sigma2_small",  x$diagnostics$sigma2_small)
      print_diag_row("sigma2_medium", x$diagnostics$sigma2_medium)
      print_diag_row("sigma2_large",  x$diagnostics$sigma2_large)
      print_diag_matrix_row("pi",     x$diagnostics$pi)
    } else {
      print_diag_matrix_row("sigma2_j", x$diagnostics$sigma2_j)
    }
  } else {
    cat("\n EM run — no MCMC diagnostics.\n")
  }

  ## Beta + GEBV --------------------------------------------------------
  cat("\n GEBV (training) quantiles\n")
  cat(" ----------------------------------------\n")
  q <- x$GEBV_summary
  cat(sprintf("   min=%s  Q1=%s  median=%s  Q3=%s  max=%s\n",
              fmt_num(q[1L]), fmt_num(q[2L]), fmt_num(q[3L]),
              fmt_num(q[4L]), fmt_num(q[5L])))

  cat("\n beta_hat\n")
  cat(" ----------------------------------------\n")
  bh <- x$beta_hat_summary
  cat(sprintf("   length=%d  nonzero=%d  mean|beta|=%s  max|beta|=%s\n",
              bh$length, bh$n_nonzero,
              fmt_num(bh$mean_abs), fmt_num(bh$max_abs)))
  if (!is.null(x$beta_samples_dim))
    cat(sprintf("   beta_samples : %d x %d (draws x alleles)\n",
                x$beta_samples_dim[1L], x$beta_samples_dim[2L]))
  if (!is.null(x$sigma2_j_dim))
    cat(sprintf("   sigma2_j_samples : %d x %d (draws x alleles)\n",
                x$sigma2_j_dim[1L], x$sigma2_j_dim[2L]))

  cat(sprintf("\n Top %d alleles by |beta_hat|\n", nrow(x$top_alleles)))
  cat(" ----------------------------------------\n")
  print(x$top_alleles, row.names = FALSE)

  cat("\n============================================================\n\n")
  invisible(x)
}

# ---- helpers --------------------------------------------------------------

#' @keywords internal
#' @noRd
fmt_num <- function(x) {
  if (is.null(x) || length(x) == 0L || !is.finite(x)) "NA"
  else formatC(x, digits = 4, format = "g")
}

#' @keywords internal
#' @noRd
print_diag_row <- function(label, d) {
  if (is.null(d)) return(invisible())
  cat(sprintf("   %-18s ESS=%8s   Z=%8s\n",
              label, fmt_num(d$ess), fmt_num(d$geweke)))
}

#' @keywords internal
#' @noRd
print_diag_matrix_row <- function(label, d) {
  if (is.null(d)) return(invisible())
  cat(sprintf("   %-18s ESS[min/med/max]=%s/%s/%s   |Z|max=%s\n",
              label,
              fmt_num(d$ess_min), fmt_num(d$ess_median), fmt_num(d$ess_max),
              fmt_num(d$geweke_absmax)))
}

#' @keywords internal
#' @noRd
print_varcomp_bayesr <- function(vc) {
  cls <- function(label, v) {
    cat(sprintf("   %-8s : mean=%s  95%% CI=[%s, %s]   pi=%s\n",
                label,
                fmt_num(v$mean), fmt_num(v$lower), fmt_num(v$upper),
                fmt_num(vc$pi[[label]])))
  }
  cat(sprintf("   %-8s :                                    pi=%s\n",
              "zero", fmt_num(vc$pi$zero)))
  cls("small",  vc$small)
  cls("medium", vc$medium)
  cls("large",  vc$large)
}

#' @keywords internal
#' @noRd
print_varcomp_bayesa <- function(vc) {
  cls <- function(label, v) {
    cat(sprintf("   %-8s : mean=%s  range=[%s, %s]  n_markers=%d\n",
                label,
                fmt_num(v$mean),
                fmt_num(v$range_lower), fmt_num(v$range_upper),
                v$n_markers))
  }
  cat("   (BayesA: per-marker variances binned into tertiles)\n")
  cls("small",  vc$small)
  cls("medium", vc$medium)
  cls("large",  vc$large)
}
