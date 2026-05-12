# R/bayesa-wrapper.R

#' Fit a BayesA Marker-Specific Variance Model (MCMC or stochastic EM)
#'
#' BayesA (Meuwissen et al., 2001) places a scaled inverse chi-squared
#' prior on each allele's effect variance, allowing heavier-tailed
#' effect-size distributions than ridge regression. Use \code{method =
#' "mcmc"} for full posterior inference or \code{method = "em"} for fast
#' stochastic-EM point estimates (Gaussian only).
#'
#' @details
#' \strong{Statistical model.}
#' The base linear model for BayesA is:
#' \deqn{y = \mathbf{1}\mu + W\beta + \varepsilon, \quad
#'   \varepsilon \sim \mathcal{N}(0,\,\sigma_e^2 I)}
#' where \eqn{y} is the phenotype vector, \eqn{\mu} is the intercept,
#' \eqn{W} is the marker design matrix (from
#' \code{\link{construct_wah_matrix}} for multi-allelic or
#' \code{\link{construct_snp_matrix}} for SNP markers), and \eqn{\beta}
#' is the allele-effect vector. When fixed effects are supplied via \code{X}:
#' \deqn{y = \mathbf{1}\mu + X\alpha + W\beta + \varepsilon}
#' \strong{BayesA prior.}
#' Each allele effect has a marker-specific variance drawn from a scaled
#' inverse chi-squared distribution (Meuwissen et al., 2001):
#' \deqn{\beta_j \mid \sigma_{\beta_j}^2 \;\sim\; \mathcal{N}(0,\,\sigma_{\beta_j}^2),
#'   \quad \sigma_{\beta_j}^2 \;\sim\; \chi^{-2}(\nu,\,S^2)}
#' where \eqn{\nu} is the degrees-of-freedom hyperparameter (\code{nu},
#' default 4.5) and \eqn{S^2 = \sigma_\beta^2 / L} is the scale, with
#' \eqn{L} denoting the total number of alleles (columns of \eqn{W}).
#' The wrapper derives \code{s_squared} as
#' \code{sigma2_g * (nu - 2) / (nu * sum_2pq)}, which sets
#' \eqn{E[\sigma_{\beta_j}^2] = \sigma_\beta^2 / \sum_j \mathrm{Var}(w_j)}.
#'
#' \strong{Heritability.}
#' \eqn{h^2} is estimated as:
#' \deqn{h^2 = \frac{\mathrm{Var}(W\hat{\beta})}
#'   {\mathrm{Var}(W\hat{\beta}) + \hat{\sigma}_e^2}}
#' This estimator is identical for SNP and multi-allelic markers,
#' treating heritability as a trait-and-population property rather than
#' a marker-panel property.
#'
#' \strong{Algorithm choice.} MCMC uses marginalised Gibbs sampling and
#' returns full posterior chains. EM is much faster but provides no
#' uncertainty quantification and does not support \code{response_type =
#' "binary"}.
#'
#' \strong{Auto-save.} By default the returned fit is saved to
#' \code{results_bayesa.Rds} in \code{getwd()}. Set \code{save_rds = FALSE}
#' for CV loops.
#'
#' \strong{Three usage scenarios.} \code{run_bayesa()} +
#' \code{summary()} + \code{predict()} support full-data fits,
#' train-test splits, and k-fold CV uniformly. See
#' \code{\link{run_bayesr}} for example loops.
#'
#' \strong{Derived parameter.} The wrapper computes
#' \code{s_squared = sigma2_g * (nu - 2) / (nu * sum(apply(w, 2, var)))}
#' so the prior expectation of each marker variance equals
#' \code{sigma2_g / sum_2pq}. This adapts to the supplied design matrix
#' and is therefore the same formula under both
#' \code{marker_type = "multiallelic"} and \code{marker_type = "snp"}.
#'
#' \strong{Marker-specific heritability note.} The \eqn{h^2} estimator in
#' the Statistical model section above is applied regardless of
#' \code{marker_type}.
#'
#' @param w Numeric design matrix (\code{n x p}). Typically the
#'   \code{$W_ah} element returned by \code{\link{construct_wah_matrix}}
#'   or the \code{$W} element returned by \code{\link{construct_snp_matrix}}.
#' @param y Phenotype vector (length \code{n}). Use 0/1 for binary traits.
#' @param wtw_diag Pre-computed \code{colSums(w^2)}.
#' @param X Optional fixed-effects design matrix (\code{n x q}). When
#'   supplied, the model is \eqn{y = X\alpha + W\beta + \mu + \epsilon}
#'   with a flat prior on \eqn{\alpha}. Do \strong{not} include a column
#'   of ones for the intercept; \eqn{\mu} is sampled separately.
#' @param marker_type One of \code{"auto"} (default), \code{"snp"}, or
#'   \code{"multiallelic"}. \code{"auto"} resolves to
#'   \code{"multiallelic"}. Setting \code{"snp"} switches the default
#'   residual-variance prior to a flat \code{InvGamma(1, 0)}
#'   (\code{a0_e = 1, b0_e = 0}); the per-marker prior scale
#'   \code{s_squared} is unaffected because it already adapts via
#'   \code{sum_2pq}. Custom supply of \code{prior_params$a0_e} is always
#'   honoured and overrides the SNP-mode flat default.
#' @param nu Degrees of freedom for the scaled inverse chi-squared prior on
#'   marker variances. Smaller values allow heavier tails. Must be > 2.
#'   Default \code{4.5}.
#' @param sigma2_g Prior total genetic variance. Typically
#'   \code{var(y) * 0.5}; for binary traits use \code{1.0}.
#' @param sigma2_e_init Initial residual variance. For binary traits fix
#'   at \code{1.0}.
#' @param prior_params Optional named list. Only \code{a0_e} is used;
#'   \code{b0_e} is derived as \code{sigma2_e_init * (a0_e - 1)}. Default
#'   \code{a0_e} is 10 for \code{marker_type = "multiallelic"} and 1 for
#'   \code{marker_type = "snp"} (with \code{b0_e = 0}); supplying
#'   \code{a0_e} explicitly always overrides the default.
#' @param mcmc_params Optional named list: \code{n_iter} (40000),
#'   \code{n_burn} (20000), \code{n_thin} (10), \code{seed} (123).
#' @param em_params Optional named list: \code{max_iter} (500), \code{tol}
#'   (\code{1e-6}).
#' @param method Either \code{"mcmc"} (default) or \code{"em"}.
#' @param response_type \code{"gaussian"} (default) or \code{"binary"}.
#'   Binary requires \code{method = "mcmc"}.
#' @param fold_id Integer label for progress messages.
#' @param save_rds If \code{TRUE}, the fit object is saved to
#'   disk as an RDS file. Set \code{FALSE} (default) for CV loops.
#' @param save_path Optional explicit RDS path. If \code{NULL} (default),
#'   defaults to \code{"results_bayesa.Rds"} in the current working
#'   directory.
#' @param verbose If \code{TRUE}, the Rust engine streams per-iteration
#'   progress (start banner, Iter X/Y diagnostics, ESS / Geweke,
#'   completion) to stderr. Default \code{FALSE} keeps the Rust side
#'   silent. The brief R post-fit summary prints regardless.
#' @param map,windsize,windnum Accepted for API symmetry with
#'   \code{\link{run_bayesr}} but \strong{not supported} for BayesA. Passing
#'   any non-NULL \code{map} raises an error; BayesA has no zero-effect
#'   mixture mass, so PIP and WPPA are ill-defined for it. Use
#'   \code{\link{run_bayesr}} for GWAS-style posterior summaries.
#'
#' @return An object of class \code{c("masbayes_bayesa", "masbayes")} — a
#'   list with the following key fields:
#' \describe{
#'   \item{\code{beta_hat, mu_hat, sigma2_e_hat, sigma2_j_hat}}{Posterior
#'     point estimates (intercept, allele effects, residual variance,
#'     per-marker variances).}
#'   \item{\code{beta_samples, sigma2_j_samples, sigma2_e_samples,
#'     mu_samples}}{Posterior chains (single-row matrices for EM).}
#'   \item{\code{GEBV / pred_train, h2, sigma2_g, sigma2_e}}{Training
#'     GEBVs (liability scale for binary), heritability, and total
#'     variance components.}
#'   \item{\code{prob_train}}{Binary only: training-set predicted
#'     probabilities \code{P(y = 1) = pnorm(pred_train)} (probit
#'     inverse link from Albert-Chib augmentation).}
#'   \item{\code{runtime}}{Elapsed seconds.}
#'   \item{\code{training_metrics}}{\code{R2}, \code{RMSE},
#'     \code{accuracy} (or \code{AUC} for binary), \code{bias}. For
#'     binary, all metrics are on the observed (probability) scale —
#'     \code{bias} is the calibration slope (1.0 = perfectly
#'     calibrated) and \code{RMSE\^2} approximates the Brier score.
#'     AUC is rank-invariant so unaffected.}
#'   \item{\code{diagnostics}}{ESS / Geweke Z (MCMC only).}
#'   \item{\code{variance_components}}{Per-marker variances binned into
#'     tertiles (small / medium / large) reporting mean, range, and
#'     marker count.}
#'   \item{\code{rds_path}}{Path of the saved RDS file, or \code{NULL}.}
#' }
#'
#' @examples
#' \dontrun{
#' d <- load_data("small")
#' mcmc <- list(n_iter = 1000L, n_burn = 500L, n_thin = 5L, seed = 123L)
#'
#' # ---- (A) SNP path -----------------------------------------------------
#' snp_train <- construct_snp_matrix(d$snp[d$train_idx, ], encoding = "zscore")
#' W_train   <- snp_train$W
#' y_train   <- d$pheno$y_cont_qtl_snp[d$train_idx]
#' X_train   <- model.matrix(~ sex - 1, data = d$pheno[d$train_idx, ])
#'
#' fit_snp <- run_bayesa(
#'   w             = W_train,
#'   y             = y_train,
#'   wtw_diag      = colSums(W_train^2),
#'   X             = X_train,
#'   marker_type   = "snp",
#'   nu            = 4.5,
#'   sigma2_g      = var(y_train) * 0.5,
#'   sigma2_e_init = var(y_train) * 0.5,
#'   mcmc_params   = mcmc,
#'   save_rds      = FALSE
#' )
#' summary(fit_snp)
#'
#' # ---- (B) Microhaplotype path -----------------------------------------
#' # d$allele_freq is the frequency table required for the training call.
#' bid    <- attr(d$mh, "block_id")
#' hap_tr <- d$mh[d$train_idx, ]
#' W_mh   <- construct_wah_matrix(hap_tr, bid, d$allele_freq)$W_ah
#' y_mh   <- d$pheno$y_cont_qtl_mh[d$train_idx]
#'
#' fit_mh <- run_bayesa(
#'   w             = W_mh,
#'   y             = y_mh,
#'   wtw_diag      = colSums(W_mh^2),
#'   X             = X_train,
#'   nu            = 4.5,
#'   sigma2_g      = var(y_mh) * 0.5,
#'   sigma2_e_init = var(y_mh) * 0.5,
#'   mcmc_params   = mcmc,
#'   save_rds      = FALSE
#' )
#' summary(fit_mh)
#' }
#'
#' @seealso \code{\link{run_bayesr}}, \code{\link{construct_wah_matrix}},
#'   \code{\link{summary.masbayes_bayesa}},
#'   \code{\link{predict.masbayes_bayesa}}
#' @references
#' Meuwissen, T. H. E., Hayes, B. J., \& Goddard, M. E. (2001).
#' Prediction of total genetic value using genome-wide dense marker
#' maps. \emph{Genetics}, 157(4), 1819-1829.
#' \doi{10.1093/genetics/157.4.1819}
#'
#' @export
run_bayesa <- function(w, y, wtw_diag,
                       X             = NULL,
                       marker_type   = c("auto", "snp", "multiallelic"),
                       nu            = 4.5,
                       sigma2_g, sigma2_e_init,
                       prior_params  = NULL,
                       mcmc_params   = NULL,
                       em_params     = NULL,
                       method        = c("mcmc", "em"),
                       response_type = c("gaussian", "binary"),
                       fold_id       = 0L,
                       save_rds      = FALSE,
                       save_path     = NULL,
                       verbose       = FALSE,
                       map           = NULL,
                       windsize      = NULL,
                       windnum       = NULL) {

  call          <- match.call()
  method        <- match.arg(method)
  response_type <- match.arg(response_type)
  marker_type   <- match.arg(marker_type)
  if (marker_type == "auto") marker_type <- "multiallelic"
  is_binary     <- response_type == "binary"

  if (is_binary && method == "em") {
    stop("response_type = 'binary' is only supported for method = 'mcmc'")
  }

  # GWAS is BayesR-only. BayesA places a continuous prior on every marker
  # variance (Scaled-Inv-Chi-Sq), so the mixture indicator that defines PIP
  # in BayesR does not exist here. Reject up front rather than silently
  # discarding the user's GWAS arguments.
  if (!is.null(map) || !is.null(windsize) || !is.null(windnum)) {
    stop("GWAS is only supported for run_bayesr(); BayesA has no ",
         "zero-effect mass required for PIP/WPPA. Use run_bayesr() instead.",
         call. = FALSE)
  }

  # FFI integer contract: Rust calls `.as_integer().unwrap()` on fold_id and
  # panics on REALSXP. Coerce here so callers may pass `fold_id = k` from a
  # numeric loop counter without the `L` suffix.
  fold_id <- as.integer(fold_id)

  if (!is.null(X)) {
    if (!is.matrix(X)) X <- as.matrix(X)
    storage.mode(X) <- "double"
    if (nrow(X) != length(y))
      stop(sprintf("nrow(X) = %d does not match length(y) = %d",
                   nrow(X), length(y)))
    if (ncol(X) == 0L) X <- NULL
  }

  sum_2pq   <- sum(apply(w, 2, var))
  s_squared <- sigma2_g * (nu - 2) / (nu * sum_2pq)

  # Track whether the caller explicitly supplied a0_e so SNP-mode does not
  # silently override an intentional user choice.
  user_supplied_a0e <- !is.null(prior_params) && !is.null(prior_params$a0_e)

  if (method == "mcmc") {
    mcmc_params <- modifyList(
      list(n_iter = 40000L, n_burn = 20000L, n_thin = 10L, seed = 123L),
      mcmc_params %||% list()
    )
    # FFI integer contract: Rust parses these via `.as_integer().unwrap()`
    # and panics on REALSXP. modifyList preserves the caller's storage mode,
    # so coerce defensively after the merge.
    mcmc_params$n_iter <- as.integer(mcmc_params$n_iter)
    mcmc_params$n_burn <- as.integer(mcmc_params$n_burn)
    mcmc_params$n_thin <- as.integer(mcmc_params$n_thin)
    mcmc_params$seed   <- as.integer(mcmc_params$seed)
    default_a0e <- if (marker_type == "snp" && !user_supplied_a0e) 1 else 10
    prior_params <- modifyList(
      list(a0_e = default_a0e),
      prior_params %||% list()
    )
    if (marker_type == "snp" && !user_supplied_a0e) {
      prior_params$b0_e <- 0
    } else {
      prior_params$b0_e <- sigma2_e_init * (prior_params$a0_e - 1)
    }

    timing <- system.time({
      raw <- run_bayesa_mcmc(w, y, wtw_diag, X, nu, s_squared,
                             sigma2_e_init, prior_params, mcmc_params,
                             fold_id, is_binary, isTRUE(verbose))
    })
  } else {
    em_params <- modifyList(
      list(max_iter = 500L, tol = 1e-6),
      em_params %||% list()
    )
    # FFI integer contract: max_iter must be INTSXP for Rust. `tol` stays
    # double (Rust uses `.as_real()`).
    em_params$max_iter <- as.integer(em_params$max_iter)

    timing <- system.time({
      raw <- run_bayesa_em(w, y, wtw_diag, X, nu, s_squared,
                           sigma2_e_init, em_params, fold_id,
                           isTRUE(verbose))
    })
  }

  fit <- finalise_fit(
    raw           = raw,
    w             = w,
    y             = y,
    model_type    = "bayesa",
    method        = method,
    response_type = response_type,
    fold_id       = fold_id,
    runtime       = timing["elapsed"],
    mcmc_params   = if (method == "mcmc") mcmc_params else NULL,
    em_params     = if (method == "em")   em_params   else NULL,
    call          = call
  )

  fit$rds_path <- maybe_save_rds(fit, save_rds, save_path)

  print_run_summary(fit)

  fit
}
