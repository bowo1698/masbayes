# R/bayesr-wrapper.R

#' Fit a BayesR Mixture Model (MCMC or stochastic EM)
#'
#' BayesR (Erbe et al., 2012) places a four-component mixture prior on
#' allele effects: \emph{zero}, \emph{small}, \emph{medium}, and
#' \emph{large} effect classes with proportions \code{pi_vec} and class
#' variances proportional to \code{sigma2_ah}. Use \code{method = "mcmc"}
#' for full Bayesian posterior inference, or \code{method = "em"} for
#' fast stochastic-EM point estimates (Gaussian only) (Wang et al., 2015).
#'
#' @details
#' \strong{Statistical model.}
#' The base linear model for BayesR is:
#' \deqn{y = \mathbf{1}\mu + W\beta + \varepsilon, \quad
#'   \varepsilon \sim \mathcal{N}(0,\,\sigma_e^2 I)}
#' where \eqn{y} is the phenotype vector, \eqn{\mu} is the intercept,
#' \eqn{W} is the marker design matrix (from
#' \code{\link{construct_wah_matrix}} for multi-allelic or
#' \code{\link{construct_snp_matrix}} for SNP markers), and \eqn{\beta}
#' is the allele-effect vector. When fixed effects are supplied via \code{X}:
#' \deqn{y = \mathbf{1}\mu + X\alpha + W\beta + \varepsilon}
#' \strong{BayesR mixture prior.}
#' Each allele effect is drawn from a four-component mixture (Erbe et al., 2012):
#' \deqn{\beta_j \mid \pi,\,\sigma_\beta^2 \;\sim\;
#'   \pi_0\,\mathcal{N}(0,0)
#'   + \pi_1\,\mathcal{N}(0,\,0.001\sigma_\beta^2)
#'   + \pi_2\,\mathcal{N}(0,\,0.01\sigma_\beta^2)
#'   + \pi_3\,\mathcal{N}(0,\,0.1\sigma_\beta^2)}
#' where \eqn{\pi = (\pi_0, \pi_1, \pi_2, \pi_3)} are mixture proportions
#' (default 0.95, 0.02, 0.02, 0.01), updated each MCMC iteration by
#' Dirichlet sampling.
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
#' returns full posterior chains; recommended when posterior uncertainty,
#' ESS, or Geweke diagnostics are needed. EM is much faster but yields no
#' uncertainty quantification and does not support \code{response_type =
#' "binary"}.
#'
#' \strong{Auto-save.} By default the returned fit is also saved to
#' \code{results_bayesr.Rds} in \code{getwd()}. Set \code{save_rds = FALSE}
#' (or pass an explicit \code{save_path}) when running cross-validation
#' loops to avoid overwriting earlier folds.
#'
#' \strong{Three usage scenarios.} The same \code{run_bayesr()} +
#' \code{summary()} + \code{predict()} pipeline supports:
#' \itemize{
#'   \item \emph{Full-data fit:} train on all data; \code{predict(fit)}
#'     returns in-sample metrics.
#'   \item \emph{Train-test split:} train on a subset; evaluate via
#'     \code{predict(fit, W_test, y_test)}.
#'   \item \emph{k-fold CV:} the user loops over folds, calling
#'     \code{run_bayesr()} with \code{save_rds = FALSE} and \code{fold_id =
#'     k}, then \code{predict()} on each held-out fold.
#' }
#'
#' \strong{Derived hyperparameters.} The wrapper computes
#' \code{b0_e = sigma2_e_init * (a0_e - 1)} and
#' \code{b0_g = sigma2_ah * (a0_g - 2) / (a0_g * (1 - pi_vec[1]))} before
#' calling the Rust engine. For EM, a \code{sigma2_vec} is derived as
#' \code{variance_class * sigma2_ah / ((1 - pi_vec[1]) * p)}.
#'
#' \strong{Marker-type-aware parameterisation.} When \code{marker_type =
#' "snp"} the wrapper rescales \code{variance_class} by
#' \code{ncol(w) / sum(apply(w, 2, var))} before passing it to the engine.
#' Combined with the engine's hardcoded division by \code{ncol(w)}, this
#' substitutes \code{sum_2pq} for \code{p} as the denominator in the
#' per-class variance, matching the alternative biallelic-SNP convention
#' where rare markers contribute proportionally to allele-frequency
#' variance. The default \code{marker_type = "multiallelic"} preserves
#' the \code{1 / p} parameterisation bit-identically.
#'
#' \strong{Marker-specific heritability note.} The \eqn{h^2} estimator in
#' the Statistical model section above is applied regardless of
#' \code{marker_type}.
#'
#' @param w Numeric design matrix (\code{n x p}). Typically the
#'   \code{$W_ah} element returned by \code{\link{construct_wah_matrix}}
#'   or the \code{$W} element returned by \code{\link{construct_snp_matrix}}.
#' @param y Phenotype vector (length \code{n}). For binary traits use 0/1
#'   coding.
#' @param wtw_diag Pre-computed \code{colSums(w^2)}, length \code{p}.
#' @param X Optional fixed-effects design matrix (\code{n x q}). When
#'   supplied, the model is \eqn{y = X\alpha + W\beta + \mu + \epsilon}
#'   with a flat prior on \eqn{\alpha}. Do \strong{not} include a column
#'   of ones for the intercept; \eqn{\mu} is sampled separately.
#' @param marker_type One of \code{"auto"} (default), \code{"snp"}, or
#'   \code{"multiallelic"}. \code{"auto"} resolves to
#'   \code{"multiallelic"}. Setting \code{"snp"} (a) rescales
#'   \code{variance_class} by \code{p / sum_2pq} so that the per-class
#'   variance denominator becomes \code{sum_2pq} instead of \code{p}, and
#'   (b) switches the default residual-variance prior to a flat
#'   \code{InvGamma(1, 0)} (\code{a0_e = 1, b0_e = 0}). Supplying
#'   \code{prior_params$a0_e} is always honoured.
#' @param pi_vec Initial mixture proportions for the four classes (zero,
#'   small, medium, large). Must sum to 1. Default
#'   \code{c(0.95, 0.02, 0.02, 0.01)}.
#' @param sigma2_e_init Initial residual variance. Typically
#'   \code{var(y) * 0.5}; for binary traits fix at \code{1.0}.
#' @param sigma2_ah Prior total genetic variance (required). Typically
#'   \code{var(y) * 0.5}; for binary traits use \code{1.0}.
#' @param sigma2_vec Optional explicit variance vector for EM; if
#'   \code{NULL} it is derived from \code{sigma2_ah}, \code{pi_vec}, and
#'   \code{prior_params$variance_class} (with the
#'   \code{marker_type = "snp"} rescaling applied when applicable).
#' @param prior_params Optional named list overriding defaults: \code{a0_e}
#'   (10 for \code{"multiallelic"}, 1 for \code{"snp"} when not supplied),
#'   \code{a0_g} (10), \code{variance_class}
#'   (\code{c(0, 0.001, 0.01, 0.1)}). \code{b0_e} and \code{b0_g} are
#'   computed internally.
#' @param mcmc_params Optional named list: \code{n_iter} (40000),
#'   \code{n_burn} (20000), \code{n_thin} (10), \code{seed} (123).
#' @param em_params Optional named list: \code{max_iter} (500), \code{tol}
#'   (\code{1e-6}).
#' @param method Either \code{"mcmc"} (default) or \code{"em"}.
#' @param response_type \code{"gaussian"} (default) or \code{"binary"}.
#'   Binary requires \code{method = "mcmc"}.
#' @param fold_id Integer label printed in progress messages; useful when
#'   running CV loops.
#' @param save_rds If \code{TRUE} (default) the fit object is written to
#'   disk as an RDS file. Set \code{FALSE} (default) for CV loops.
#' @param save_path Optional explicit RDS path. If \code{NULL} (default),
#'   defaults to \code{"results_bayesr.Rds"} in the current working
#'   directory.
#' @param verbose If \code{TRUE}, the Rust engine streams per-iteration
#'   progress (start banner, Iter X/Y diagnostics, ESS / Geweke,
#'   completion) to stderr. Default \code{FALSE} keeps the Rust side
#'   silent. The brief R post-fit summary (model, runtime, \eqn{h^2},
#'   ESS, Geweke) prints regardless.
#' @param map Optional GWAS map. Supplying \code{map} (together with
#'   exactly one of \code{windsize} or \code{windnum}) computes per-allele
#'   PIP, per-block PIP, and per-window WPPA from the posterior samples
#'   and attaches them to the fit. For SNP markers, supply a data.frame
#'   with columns \code{SNP}, \code{CHROM}, \code{POS} (one row per W
#'   column). For multi-allelic (MH) markers, supply a data.frame with
#'   columns \code{block_id}, \code{chr}, \code{start_pos}, \code{end_pos}
#'   (one row per unique block_id in \code{attr(w, "block_id")}); this
#'   matches the schema of \code{microhaplotype_coordinates.csv} from the
#'   \code{maspipeline} tool. Marker type is auto-detected from
#'   \code{attr(w, "block_id")}. \code{map = NULL} (default) leaves the
#'   fit object byte-identical to v1.4.0 output.
#' @param windsize Window size in base pairs for WPPA. Mutually exclusive
#'   with \code{windnum}. Required when \code{map} is supplied unless
#'   \code{windnum} is.
#' @param windnum Number of consecutive markers (or MH blocks) per window
#'   for WPPA. Mutually exclusive with \code{windsize}.
#'
#' @return An object of class \code{c("masbayes_bayesr", "masbayes")} — a
#'   list with the following key fields:
#' \describe{
#'   \item{\code{beta_hat, mu_hat, sigma2_e_hat}}{Posterior point
#'     estimates.}
#'   \item{\code{beta_samples, gamma_samples, pi_samples,
#'     sigma2_e_samples, sigma2_small_samples, sigma2_medium_samples,
#'     sigma2_large_samples, mu_samples}}{Posterior chains (single-row
#'     matrices for EM).}
#'   \item{\code{GEBV / pred_train}}{Training genomic estimated breeding
#'     values. For binary traits this is on the \emph{liability} scale.}
#'   \item{\code{prob_train}}{Binary only: training-set predicted
#'     probabilities \code{P(y = 1) = pnorm(pred_train)} (probit
#'     inverse link from Albert-Chib augmentation).}
#'   \item{\code{h2, sigma2_g, sigma2_e}}{Heritability and variance
#'     components.}
#'   \item{\code{runtime}}{Elapsed seconds.}
#'   \item{\code{training_metrics}}{\code{R2}, \code{RMSE},
#'     \code{accuracy} (or \code{AUC} for binary), \code{bias}. For
#'     binary, all metrics are on the observed (probability) scale —
#'     \code{bias} is the calibration slope (1.0 = perfectly
#'     calibrated) and \code{RMSE\^2} approximates the Brier score.
#'     AUC is rank-invariant so unaffected.}
#'   \item{\code{diagnostics}}{ESS and Geweke Z for key parameters
#'     (MCMC only).}
#'   \item{\code{variance_components}}{Posterior mean and 95\% CI for
#'     each class plus the mixture proportions \code{pi}.}
#'   \item{\code{rds_path}}{Path of the saved RDS file, or \code{NULL}
#'     when \code{save_rds = FALSE}.}
#'   \item{\code{pip}}{(GWAS only, since v0.5.0) Per-allele posterior
#'     inclusion probability, length \code{ncol(w)}. Computed as
#'     \code{colMeans(gamma_samples != 0)}. \code{NULL} unless \code{map}
#'     was supplied.}
#'   \item{\code{pip_block}}{(GWAS only) Per-block PIP, length
#'     \code{nrow(map)}. For SNP markers, identical to \code{pip}. For
#'     multi-allelic markers, a block is "active" in a posterior sample
#'     if at least one of its alleles has \code{gamma != 0}. \code{NULL}
#'     unless \code{map} was supplied.}
#'   \item{\code{gwas}}{(GWAS only) Data frame with columns \code{Wind},
#'     \code{Chr}, \code{N}, \code{Start}, \code{End}, \code{WPPA}. Each
#'     row is one physical window; \code{WPPA} is the window posterior
#'     probability of association, i.e. the fraction of posterior samples
#'     in which at least one marker / block in the window has non-zero
#'     effect.}
#'   \item{\code{gwas_meta}}{(GWAS only) List capturing the call:
#'     \code{windsize}, \code{windnum}, \code{midpoint_convention},
#'     \code{marker_type}, \code{n_windows}.}
#' }
#'
#' @section GWAS:
#' Supplying a \code{map} together with exactly one of \code{windsize} or
#' \code{windnum} turns \code{run_bayesr()} into a one-shot GWAS engine
#' alongside genomic prediction. The Rust kernel is untouched; PIP and
#' WPPA are derived from the existing \code{gamma_samples} matrix.
#'
#' \strong{Map schema.} For \strong{SNP} markers, \code{map} is a
#' \code{data.frame} with columns \code{SNP} (character), \code{CHROM}
#' (integer), \code{POS} (integer), one row per W column.
#' For \strong{multi-allelic (MH)} markers, \code{map} has columns
#' \code{block_id} (character), \code{chr} (integer), \code{start_pos}
#' (integer), \code{end_pos} (integer), one row per unique block_id in
#' \code{attr(w, "block_id")}. The MH schema matches the
#' \code{microhaplotype_coordinates.csv} produced by the
#' \code{maspipeline} tool, so production pipelines can pass it through
#' without translation.
#'
#' \strong{Windows.} \code{windsize} (bp) closes a window when the next
#' marker's position exceeds the window start by \code{windsize}.
#' \code{windnum} groups \code{windnum} consecutive markers / blocks per
#' window. Windows never cross chromosome boundaries.
#'
#' \strong{Constraints.} GWAS requires \code{method = "mcmc"} (EM has no
#' posterior samples). \code{\link{run_bayesa}} accepts the same
#' arguments for API symmetry but errors out (BayesA has no zero-effect
#' mixture mass, so PIP / WPPA are ill-defined). \code{map = NULL}
#' (default) leaves the fit object byte-identical to pre-v0.5.0 output.
#'
#' @examples
#' \dontrun{
#' d <- load_data("small")
#' mcmc <- list(n_iter = 1000L, n_burn = 500L, n_thin = 5L, seed = 123L)
#'
#' # ---- (A) SNP path -----------------------------------------------------
#' # Pair `encoding = "zscore"` on the matrix builder with
#' # `marker_type = "snp"` on the fitter for the alternative biallelic-SNP
#' # convention.
#' snp_train <- construct_snp_matrix(d$snp[d$train_idx, ], encoding = "zscore")
#' W_train   <- snp_train$W
#' y_train   <- d$pheno$y_cont_qtl_snp[d$train_idx]
#' X_train   <- model.matrix(~ sex - 1, data = d$pheno[d$train_idx, ])
#'
#' fit_snp <- run_bayesr(
#'   w             = W_train,
#'   y             = y_train,
#'   wtw_diag      = colSums(W_train^2),
#'   X             = X_train,
#'   marker_type   = "snp",
#'   sigma2_e_init = var(y_train) * 0.5,
#'   sigma2_ah     = var(y_train) * 0.5,
#'   mcmc_params   = mcmc,
#'   save_rds      = FALSE
#' )
#' summary(fit_snp)
#'
#' # ---- (B) Microhaplotype path -----------------------------------------
#' # d$mh is consumable directly by construct_wah_matrix via attr("block_id").
#' # d$allele_freq is the frequency table required for the training call.
#' bid    <- attr(d$mh, "block_id")
#' hap_tr <- d$mh[d$train_idx, ]
#' W_mh   <- construct_wah_matrix(hap_tr, bid, d$allele_freq)$W_ah
#' y_mh   <- d$pheno$y_cont_qtl_mh[d$train_idx]
#'
#' # ---- (C) GWAS (BayesR-only) ------------------------------------------
#' # Supply `map` together with `windsize` (bp) or `windnum` (markers per
#' # window) to compute per-allele PIP, per-block PIP, and per-window WPPA.
#' # Schema for SNP map: SNP / CHROM / POS (one row per W column).
#' # The map's row order does not need to be pre-sorted.
#' map_snp <- data.frame(
#'   SNP   = paste0("M", seq_len(ncol(W_train))),
#'   CHROM = rep(seq_len(4L), each = ncol(W_train) %/% 4L),
#'   POS   = rep(seq(1e6, by = 1e5,
#'                   length.out = ncol(W_train) %/% 4L), 4L)
#' )
#' fit_gwas <- run_bayesr(
#'   w             = W_train,
#'   y             = y_train,
#'   wtw_diag      = colSums(W_train^2),
#'   marker_type   = "snp",
#'   sigma2_e_init = var(y_train) * 0.5,
#'   sigma2_ah     = var(y_train) * 0.5,
#'   mcmc_params   = mcmc,
#'   save_rds      = FALSE,
#'   map           = map_snp,
#'   windsize      = 5e5
#' )
#' head(fit_gwas$gwas[order(-fit_gwas$gwas$WPPA), ], 5)
#'
#' fit_mh <- run_bayesr(
#'   w             = W_mh,
#'   y             = y_mh,
#'   wtw_diag      = colSums(W_mh^2),
#'   X             = X_train,
#'   sigma2_e_init = var(y_mh) * 0.5,
#'   sigma2_ah     = var(y_mh) * 0.5,
#'   mcmc_params   = mcmc,
#'   save_rds      = FALSE
#' )
#' summary(fit_mh)
#' }
#'
#' @seealso \code{\link{run_bayesa}}, \code{\link{construct_wah_matrix}},
#'   \code{\link{summary.masbayes_bayesr}},
#'   \code{\link{predict.masbayes_bayesr}}
#' @references
#' Erbe, M., Hayes, B. J., Matukumalli, L. K., Goswami, S., Bowman, P. J.,
#' Reich, C. M., Mason, B. A., \& Goddard, M. E. (2012). Improving
#' accuracy of genomic predictions within and between dairy cattle
#' breeds with imputed high-density single nucleotide polymorphism
#' panels. \emph{Journal of Dairy Science}, 95(7), 4114-4129.
#' \doi{10.3168/jds.2011-5019}
#'
#' Wang, T., Chen, YP. P., \& Goddard, M. E. (2015).
#' A computationally efficient algorithm for genomic prediction using
#' a Bayesian model. \emph{Genet Sel Evol}, 47(34).
#' \doi{10.1186/s12711-014-0082-4}
#'
#' @export
run_bayesr <- function(w, y, wtw_diag,
                       X             = NULL,
                       marker_type   = c("auto", "snp", "multiallelic"),
                       pi_vec        = c(0.95, 0.02, 0.02, 0.01),
                       sigma2_e_init,
                       sigma2_ah     = NULL,
                       sigma2_vec    = NULL,
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

  if (!is.null(X)) {
    if (!is.matrix(X)) X <- as.matrix(X)
    storage.mode(X) <- "double"
    if (nrow(X) != length(y))
      stop(sprintf("nrow(X) = %d does not match length(y) = %d",
                   nrow(X), length(y)))
    if (ncol(X) == 0L) X <- NULL
  }

  call          <- match.call()
  method        <- match.arg(method)
  response_type <- match.arg(response_type)
  marker_type   <- match.arg(marker_type)
  if (marker_type == "auto") marker_type <- "multiallelic"
  is_binary     <- response_type == "binary"

  if (is_binary && method == "em") {
    stop("response_type = 'binary' is only supported for method = 'mcmc'")
  }
  if (is.null(sigma2_ah)) stop("sigma2_ah required for both MCMC and EM")

  # GWAS argument validation runs BEFORE the MCMC so a malformed map fails
  # fast (within milliseconds) instead of after a long run.
  validated_map <- NULL
  if (!is.null(map)) {
    if (method == "em") {
      stop("GWAS requires MCMC (EM produces no posterior samples). ",
           "Use method = 'mcmc' when supplying map.", call. = FALSE)
    }
    if (!is.null(windsize) && !is.null(windnum)) {
      stop("Specify exactly one of windsize / windnum", call. = FALSE)
    }
    if (is.null(windsize) && is.null(windnum)) {
      stop("Specify windsize (bp) or windnum (markers per window) ",
           "when map is supplied", call. = FALSE)
    }
    validated_map <- .validate_map(map, w)
  }

  # FFI integer contract: Rust calls `.as_integer().unwrap()` on fold_id and
  # panics on REALSXP. Coerce here so callers may pass `fold_id = k` from a
  # numeric loop counter without the `L` suffix.
  fold_id <- as.integer(fold_id)

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
      list(a0_e = default_a0e, a0_g = 10,
           variance_class = c(0, 0.001, 0.01, 0.1)),
      prior_params %||% list()
    )
    if (marker_type == "snp") {
      sum_2pq <- sum(apply(w, 2, var))
      prior_params$variance_class <- prior_params$variance_class *
                                     (ncol(w) / sum_2pq)
    }
    if (marker_type == "snp" && !user_supplied_a0e) {
      prior_params$b0_e <- 0
    } else {
      prior_params$b0_e <- sigma2_e_init * (prior_params$a0_e - 1)
    }
    prior_params$b0_g <- sigma2_ah * (prior_params$a0_g - 2) /
                        prior_params$a0_g / (1 - pi_vec[1])

    timing <- system.time({
      raw <- run_bayesr_mcmc(w, y, wtw_diag, X, pi_vec,
                             sigma2_e_init, sigma2_ah, prior_params,
                             mcmc_params, fold_id, is_binary,
                             isTRUE(verbose))
    })
  } else {
    em_params <- modifyList(
      list(max_iter = 500L, tol = 1e-6),
      em_params %||% list()
    )
    # FFI integer contract: max_iter must be INTSXP for Rust. `tol` stays
    # double (Rust uses `.as_real()`).
    em_params$max_iter <- as.integer(em_params$max_iter)
    variance_class <- prior_params$variance_class %||% c(0, 0.001, 0.01, 0.1)
    if (marker_type == "snp") {
      sum_2pq        <- sum(apply(w, 2, var))
      variance_class <- variance_class * (ncol(w) / sum_2pq)
    }
    varg_init      <- as.numeric(sigma2_ah / ((1 - pi_vec[1]) * ncol(w)))
    sigma2_vec     <- as.vector(variance_class) * varg_init
    sigma2_vec[1]  <- 0

    timing <- system.time({
      raw <- run_bayesr_em(w, y, wtw_diag, X, pi_vec, sigma2_vec,
                           sigma2_e_init, em_params, fold_id,
                           isTRUE(verbose))
    })
  }

  fit <- finalise_fit(
    raw           = raw,
    w             = w,
    y             = y,
    model_type    = "bayesr",
    method        = method,
    response_type = response_type,
    fold_id       = fold_id,
    runtime       = timing["elapsed"],
    mcmc_params   = if (method == "mcmc") mcmc_params else NULL,
    em_params     = if (method == "em")   em_params   else NULL,
    call          = call,
    map           = validated_map,
    windsize      = windsize,
    windnum       = windnum,
    marker_type   = marker_type
  )

  fit$rds_path <- maybe_save_rds(fit, save_rds, save_path)

  print_run_summary(fit)

  fit
}
