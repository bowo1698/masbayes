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

#' Binary AUC via the Wilcoxon-Mann-Whitney U statistic.
#'
#' Empirical ROC AUC equals \code{U / (n_pos * n_neg)} where U is the
#' Mann-Whitney U statistic with mid-rank tie handling. This is
#' mathematically identical to \code{pROC::auc(pROC::roc(...,
#' direction = "<"))} on empirical (non-smoothed) curves, but uses only
#' base R \code{rank()} — no Rcpp / pROC dependency chain.
#'
#' Returns NA when either class is empty (degenerate ROC).
#' @keywords internal
#' @noRd
.compute_auc <- function(y_bin, yhat) {
  pos <- yhat[y_bin == 1L]
  neg <- yhat[y_bin == 0L]
  if (length(pos) == 0L || length(neg) == 0L) return(NA_real_)
  ranks <- rank(c(pos, neg), ties.method = "average")
  (sum(ranks[seq_along(pos)]) - length(pos) * (length(pos) + 1L) / 2L) /
    (length(pos) * length(neg))
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
    auc <- .compute_auc(y_bin, as.numeric(yhat))
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
#'
#' New GWAS args (v0.5.0) are NULL-default and only active for BayesR. When
#' `map` is NULL the function's behaviour and output are byte-identical to
#' v1.4.0 (hard backward-compat invariant).
#' @keywords internal
#' @noRd
finalise_fit <- function(raw, w, y, model_type, method, response_type,
                         fold_id, runtime, mcmc_params, em_params, call,
                         map = NULL, windsize = NULL, windnum = NULL,
                         marker_type = NULL) {
  is_binary <- identical(response_type, "binary")

  yhat_train <- as.numeric(raw$pred_train)
  if (is_binary) {
    # pnorm(eta) maps liability to probability for binary metrics.
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

  # GWAS branch attaches pip/gwas/gwas_meta when caller supplied `map`.
  if (!is.null(map)) {
    if (!identical(model_type, "bayesr")) {
      stop("GWAS branch is only valid for BayesR (internal contract violation)",
           call. = FALSE)
    }
    map_marker_type <- attr(map, "marker_type") %||% marker_type
    unit_index <- .resolve_units(w, map)
    raw$pip        <- .compute_pip(raw$gamma_samples)
    raw$pip_block  <- .compute_pip_block(raw$gamma_samples,
                                         unit_index, nrow(map))
    win_info       <- .cut_windows(map, windsize, windnum, map_marker_type)
    raw$gwas       <- .compute_wppa(raw$gamma_samples, unit_index,
                                    win_info$windows, win_info$window_id)
    raw$gwas_meta  <- list(
      windsize = windsize,
      windnum  = windnum,
      midpoint_convention = if (map_marker_type == "snp")
        "POS" else "(start_pos+end_pos)/2",
      marker_type = map_marker_type,
      n_windows   = nrow(raw$gwas)
    )
  }

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

# GWAS helpers: per-allele PIP, per-block PIP, per-window WPPA from
# BayesR posterior samples. Called only when run_bayesr() receives `map`.

#' Validate the user-supplied map data.frame against W.
#'
#' Detects marker type from `attr(W, "block_id")`: present → multi-allelic
#' (MH), absent → SNP. Returns the map after coercion to canonical types
#' with `attr(map, "marker_type")` set. Errors enumerate the diff so the
#' user can fix the input directly.
#' @keywords internal
#' @noRd
.validate_map <- function(map, W) {
  if (!is.data.frame(map)) {
    stop("map must be a data.frame; got ", class(map)[1L], call. = FALSE)
  }
  block_id_attr <- attr(W, "block_id")
  marker_type   <- if (is.null(block_id_attr)) "snp" else "multiallelic"

  if (marker_type == "snp") {
    required <- c("SNP", "CHROM", "POS")
    missing_cols <- setdiff(required, names(map))
    if (length(missing_cols) > 0L) {
      stop("map is missing required columns for SNP path: ",
           paste(missing_cols, collapse = ", "),
           " (expected: ", paste(required, collapse = ", "), ")",
           call. = FALSE)
    }
    if (nrow(map) != ncol(W)) {
      stop("SNP map has ", nrow(map), " rows but W has ", ncol(W),
           " columns; one row per W column is required.", call. = FALSE)
    }
    out <- map
    out$SNP   <- as.character(out$SNP)
    out$CHROM <- as.integer(out$CHROM)
    out$POS   <- as.integer(out$POS)
  } else {
    required <- c("block_id", "chr", "start_pos", "end_pos")
    missing_cols <- setdiff(required, names(map))
    if (length(missing_cols) > 0L) {
      stop("map is missing required columns for MH path: ",
           paste(missing_cols, collapse = ", "),
           " (expected: ", paste(required, collapse = ", "), ")",
           call. = FALSE)
    }
    blocks_W   <- unique(as.character(block_id_attr))
    blocks_map <- unique(as.character(map$block_id))
    missing_in_map <- setdiff(blocks_W, blocks_map)
    extra_in_map   <- setdiff(blocks_map, blocks_W)
    if (length(missing_in_map) > 0L || length(extra_in_map) > 0L) {
      # Show up to first 5 of each to keep error readable
      parts <- character()
      if (length(missing_in_map) > 0L) {
        parts <- c(parts, sprintf(
          "missing in map: %s%s",
          paste(utils::head(missing_in_map, 5L), collapse = ", "),
          if (length(missing_in_map) > 5L)
            sprintf(" (and %d more)", length(missing_in_map) - 5L) else ""))
      }
      if (length(extra_in_map) > 0L) {
        parts <- c(parts, sprintf(
          "extra in map (not in attr(W, 'block_id')): %s%s",
          paste(utils::head(extra_in_map, 5L), collapse = ", "),
          if (length(extra_in_map) > 5L)
            sprintf(" (and %d more)", length(extra_in_map) - 5L) else ""))
      }
      stop("map$block_id does not match attr(W, 'block_id'): ",
           paste(parts, collapse = "; "), call. = FALSE)
    }
    if (length(blocks_map) != nrow(map)) {
      stop("map$block_id contains duplicates; ",
           "one row per unique block is required.", call. = FALSE)
    }
    out <- map
    out$block_id  <- as.character(out$block_id)
    out$chr       <- as.integer(out$chr)
    out$start_pos <- as.integer(out$start_pos)
    out$end_pos   <- as.integer(out$end_pos)
  }
  attr(out, "marker_type") <- marker_type
  out
}

#' Map each W column to a map row (unit index).
#'
#' SNP path: identity (`1:ncol(W)`). MH path: `match()` block_id attribute
#' against `map$block_id`. This is the single point of SNP/MH asymmetry —
#' downstream helpers consume the resulting integer vector uniformly.
#' @keywords internal
#' @noRd
.resolve_units <- function(W, map) {
  marker_type <- attr(map, "marker_type") %||% "snp"
  if (marker_type == "snp") {
    return(seq_len(ncol(W)))
  }
  block_id_attr <- attr(W, "block_id")
  if (is.null(block_id_attr) || length(block_id_attr) != ncol(W)) {
    stop("attr(W, 'block_id') is missing or has wrong length for MH path",
         call. = FALSE)
  }
  unit_index <- match(as.character(block_id_attr),
                      as.character(map$block_id))
  if (anyNA(unit_index)) {
    # Should have been caught by .validate_map; defensive sentinel.
    stop("Some W columns have block_ids not present in map (internal error)",
         call. = FALSE)
  }
  unit_index
}

#' Per-allele posterior inclusion probability.
#'
#' For each W column j: fraction of post-burn-in samples in which the
#' BayesR mixture indicator `gamma_j` is non-zero. Length `ncol(W)`.
#' @keywords internal
#' @noRd
.compute_pip <- function(gamma_samples) {
  if (is.null(gamma_samples) || !is.matrix(gamma_samples)) {
    stop("gamma_samples must be a matrix (n_save x p) for PIP computation",
         call. = FALSE)
  }
  colMeans(gamma_samples != 0)
}

#' Per-block posterior inclusion probability.
#'
#' A block is "active" in a sample if AT LEAST ONE allele in the block has
#' `gamma != 0` in that sample. Returns length `n_units`. SNP fast path
#' short-circuits when each W column is its own block (n_units == ncol(W)).
#' @keywords internal
#' @noRd
.compute_pip_block <- function(gamma_samples, unit_index, n_units) {
  active <- gamma_samples != 0
  if (n_units == ncol(active)) {
    # SNP: each column == its own block
    return(colMeans(active))
  }
  n_save <- nrow(active)
  block_active <- vapply(seq_len(n_units), function(b) {
    cols <- which(unit_index == b)
    if (length(cols) == 0L) {
      # No W column for this block (validation would normally catch);
      # treat as never-active.
      return(rep(FALSE, n_save))
    }
    rowSums(active[, cols, drop = FALSE]) > 0L
  }, logical(n_save))
  colMeans(block_active)
}

#' Cut units into windows by physical position.
#'
#' Per-chromosome window cutting; windows never cross chromosome
#' boundaries. In `windsize` mode a window closes when the next unit's
#' position exceeds `window_start + windsize`. In `windnum` mode each
#' chromosome is partitioned into chunks of `windnum` consecutive units
#' (sorted by physical position). MH midpoint = `(start_pos + end_pos) %/% 2L`;
#' SNP position = `POS`.
#'
#' Returns: list(windows = data.frame, window_id = integer vector aligned
#' to map rows). Window ids are consecutive integers 1..n_windows; no gaps.
#' @keywords internal
#' @noRd
.cut_windows <- function(map, windsize, windnum, marker_type) {
  if (is.null(windsize) && is.null(windnum)) {
    stop("Specify windsize (bp) or windnum (markers per window) ",
         "when map is supplied", call. = FALSE)
  }
  if (!is.null(windsize) && !is.null(windnum)) {
    stop("Specify exactly one of windsize / windnum", call. = FALSE)
  }

  n_units <- nrow(map)
  if (marker_type == "snp") {
    pos <- as.integer(map$POS)
    chr <- as.integer(map$CHROM)
  } else {
    # Integer midpoint via %/% (start_pos / end_pos non-negative).
    pos <- as.integer(
      (as.numeric(map$start_pos) + as.numeric(map$end_pos)) %/% 2L)
    chr <- as.integer(map$chr)
  }

  window_id <- integer(n_units)
  unique_chrs <- sort(unique(chr))
  next_wid <- 1L

  for (cur_chr in unique_chrs) {
    in_chr_idx <- which(chr == cur_chr)
    # Sort within-chr by physical position; window_id is then assigned in
    # position order, but written back through `ord` to preserve the
    # original map row alignment.
    ord     <- in_chr_idx[order(pos[in_chr_idx])]
    pos_chr <- pos[ord]
    n_chr   <- length(ord)
    chr_wid <- integer(n_chr)

    if (!is.null(windsize)) {
      wsize      <- as.numeric(windsize)
      wstart_pos <- pos_chr[1L]
      cur_w      <- next_wid
      for (i in seq_len(n_chr)) {
        if (pos_chr[i] - wstart_pos > wsize) {
          cur_w      <- cur_w + 1L
          wstart_pos <- pos_chr[i]
        }
        chr_wid[i] <- cur_w
      }
      next_wid <- cur_w + 1L
    } else {
      wnum     <- as.integer(windnum)
      chunk    <- ((seq_len(n_chr) - 1L) %/% wnum) + 1L
      chr_wid  <- chunk + next_wid - 1L
      next_wid <- max(chr_wid) + 1L
    }

    window_id[ord] <- chr_wid
  }

  unique_wids <- sort(unique(window_id))
  windows <- data.frame(
    Wind  = sprintf("wind%d", unique_wids),
    Chr   = vapply(unique_wids,
                   function(w) chr[window_id == w][1L], integer(1L)),
    N     = vapply(unique_wids,
                   function(w) sum(window_id == w), integer(1L)),
    Start = vapply(unique_wids,
                   function(w) min(pos[window_id == w]), integer(1L)),
    End   = vapply(unique_wids,
                   function(w) max(pos[window_id == w]), integer(1L)),
    stringsAsFactors = FALSE
  )

  list(windows = windows, window_id = window_id)
}

#' Window posterior probability of association (WPPA).
#'
#' For each window: fraction of post-burn-in samples in which at least one
#' unit (SNP or MH block) inside the window has `gamma != 0`. Returns the
#' input `windows` data.frame with a new `WPPA` column appended.
#'
#' The expensive step is materialising `unit_active` (n_save x n_units);
#' the SNP fast path reuses `active` directly. Per-window aggregation is
#' a single `rowSums` per window over the cached `unit_active`.
#' @keywords internal
#' @noRd
.compute_wppa <- function(gamma_samples, unit_index, windows, window_id) {
  active <- gamma_samples != 0
  n_save <- nrow(active)
  n_units <- length(window_id)

  if (n_units == ncol(active)) {
    unit_active <- active
  } else {
    unit_active <- vapply(seq_len(n_units), function(b) {
      cols <- which(unit_index == b)
      if (length(cols) == 0L) return(rep(FALSE, n_save))
      rowSums(active[, cols, drop = FALSE]) > 0L
    }, logical(n_save))
  }

  unique_wids <- sort(unique(window_id))
  WPPA <- vapply(unique_wids, function(w) {
    units_in_w <- which(window_id == w)
    mean(rowSums(unit_active[, units_in_w, drop = FALSE]) > 0L)
  }, numeric(1L))

  out <- windows
  out$WPPA <- WPPA
  out
}
