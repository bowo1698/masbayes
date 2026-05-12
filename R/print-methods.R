# R/print-methods.R
# Brief print method for masbayes fit objects (used when a user types `fit`
# at the console). Detailed reporting lives in summary().

#' Print a masbayes fit (brief, S3 dispatch).
#' @noRd
#' @export
print.masbayes <- function(x, ...) {
  model_label <- switch(x$model_type,
                        bayesr = "BayesR", bayesa = "BayesA", "masbayes")
  algo        <- toupper(x$method)
  trait_tag   <- if (identical(x$response_type, "binary")) " (binary trait)" else ""

  cat(sprintf("<masbayes fit: %s with %s%s>\n",
              model_label, algo, trait_tag))
  cat(sprintf("  n = %d, p = %d, runtime = %.2fs\n",
              x$n, x$p, x$runtime))
  cat(sprintf("  h2 = %.3f, sigma2_g = %.3f, sigma2_e = %.3f\n",
              x$h2, x$sigma2_g, x$sigma2_e))
  if (!is.null(x$rds_path))
    cat(sprintf("  saved: %s\n", x$rds_path))
  # One-line GWAS indicator — only present when run_bayesr() was called
  # with a `map` argument (BayesR-only path).
  if (!is.null(x$gwas)) {
    cat(sprintf("  GWAS: %d windows, top WPPA = %.3f\n",
                nrow(x$gwas), max(x$gwas$WPPA)))
  }
  cat("  Use summary(x) for full report; ",
      "predict(x, newdata, y_new) to evaluate.\n", sep = "")
  invisible(x)
}
