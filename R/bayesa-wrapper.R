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
#' \deqn{\beta_j \mid \sigma_{\beta_j}^2 \;\sim\; \mathcal{N}(0,\,\sigma_{\beta_j}^2),
#'   \quad \sigma_{\beta_j}^2 \;\sim\; \chi^{-2}(\nu,\,S^2)}
#' Residual: \eqn{\sigma_e^2 \sim \mathrm{InvGamma}(a_{0e},\,b_{0e})}.
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
#'   \code{"multiallelic"}. Selects the default prior for the BayesA
#'   sampler — see \code{nu}, \code{sigma2_g}, \code{sigma2_e_init},
#'   \code{prior_params}.
#' @param nu Degrees of freedom for the scaled inverse chi-squared prior
#'   on marker variances. Smaller values give heavier tails. Must be > 2.
#'   Default: \code{4} for \code{"snp"}, \code{4.5} for \code{"multiallelic"}.
#' @param sigma2_g Prior total genetic variance. Default for
#'   \code{"snp"}: \code{var(y) / 4}. \code{"multiallelic"}: required
#'   (typically \code{var(y) * 0.5}; for binary use \code{1.0}).
#' @param sigma2_e_init Initial residual variance. Default for
#'   \code{"snp"}: \code{var(y) / 2}. \code{"multiallelic"}: required
#'   (typically \code{var(y) * 0.5}; for binary use \code{1.0}).
#' @param prior_params Optional named list. Only \code{a0_e} is used;
#'   \code{b0_e} is computed as \code{sigma2_e_init * (a0_e - 1)},
#'   except when SNP-mode flat default applies (\code{a0_e = -1},
#'   \code{b0_e = 0}). User-supplied \code{a0_e} always wins. Default
#'   \code{a0_e}: \code{-1} for \code{"snp"}, \code{10} for
#'   \code{"multiallelic"}.
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

  # GWAS requires a zero-effect mixture mass (PIP); BayesA has no such mass.
  if (!is.null(map) || !is.null(windsize) || !is.null(windnum)) {
    stop("GWAS is only supported for run_bayesr(); BayesA has no ",
         "zero-effect mass required for PIP/WPPA. Use run_bayesr() instead.",
         call. = FALSE)
  }

  # Rust FFI expects INTSXP for fold_id.
  fold_id <- as.integer(fold_id)

  if (!is.null(X)) {
    if (!is.matrix(X)) X <- as.matrix(X)
    storage.mode(X) <- "double"
    if (nrow(X) != length(y))
      stop(sprintf("nrow(X) = %d does not match length(y) = %d",
                   nrow(X), length(y)))
    if (ncol(X) == 0L) X <- NULL
  }

  # Track explicit supply so SNP-mode defaults only fill unsupplied fields.
  user_supplied_a0e       <- !is.null(prior_params) && !is.null(prior_params$a0_e)
  user_supplied_nu        <- !missing(nu) && !is.null(nu)
  user_supplied_sigma2_g  <- !missing(sigma2_g) && !is.null(sigma2_g)
  user_supplied_sigma2_e0 <- !missing(sigma2_e_init) && !is.null(sigma2_e_init)

  # SNP-mode init defaults from h2=0.5, dfvg=4:
  #   nu = 4,  sigma2_g = vary/4,  sigma2_e_init = vary/2
  vary <- var(y)
  if (marker_type == "snp") {
    if (!user_supplied_nu)        nu            <- 4
    if (!user_supplied_sigma2_g)  sigma2_g      <- vary / 4
    if (!user_supplied_sigma2_e0) sigma2_e_init <- vary / 2
  }
  if (missing(sigma2_g) || is.null(sigma2_g)) {
    stop("sigma2_g required for both MCMC and EM")
  }
  if (missing(sigma2_e_init) || is.null(sigma2_e_init)) {
    stop("sigma2_e_init required for both MCMC and EM")
  }

  # Per-marker prior scale: s_squared = sigma2_g * (nu - 2) / (nu * sum_2pq)
  sum_2pq   <- sum(apply(w, 2, var))
  s_squared <- sigma2_g * (nu - 2) / (nu * sum_2pq)

  if (method == "mcmc") {
    mcmc_params <- modifyList(
      list(n_iter = 40000L, n_burn = 20000L, n_thin = 10L, seed = 123L),
      mcmc_params %||% list()
    )
    # Rust FFI expects INTSXP for iter/burn/thin/seed.
    mcmc_params$n_iter <- as.integer(mcmc_params$n_iter)
    mcmc_params$n_burn <- as.integer(mcmc_params$n_burn)
    mcmc_params$n_thin <- as.integer(mcmc_params$n_thin)
    mcmc_params$seed   <- as.integer(mcmc_params$seed)
    # SNP-mode a0_e default = -1 (improper flat on sigma^2_e).
    default_a0e <- if (marker_type == "snp" && !user_supplied_a0e) -1 else 10
    prior_params <- modifyList(
      list(a0_e = default_a0e),
      prior_params %||% list()
    )
    if (marker_type == "snp" && !user_supplied_a0e) {
      prior_params$b0_e <- 0
    } else {
      prior_params$b0_e <- sigma2_e_init * (prior_params$a0_e - 1)
    }

    # marker_type = "snp" routes to the dedicated SNP kernel (gaussian +
    # binary via Albert-Chib); everything else uses the multiallelic kernel.
    use_snp_kernel <- identical(marker_type, "snp")
    timing <- system.time({
      raw <- if (use_snp_kernel) {
        run_bayesa_snp_mcmc(w, y, wtw_diag, X, nu, s_squared,
                            sigma2_e_init, prior_params, mcmc_params,
                            fold_id, is_binary, isTRUE(verbose))
      } else {
        run_bayesa_mcmc(w, y, wtw_diag, X, nu, s_squared,
                        sigma2_e_init, prior_params, mcmc_params,
                        fold_id, is_binary, isTRUE(verbose))
      }
    })
  } else {
    em_params <- modifyList(
      list(max_iter = 500L, tol = 1e-6),
      em_params %||% list()
    )
    # Rust FFI expects INTSXP for max_iter (tol stays double).
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
