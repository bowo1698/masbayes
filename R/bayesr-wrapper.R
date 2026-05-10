# R/bayesr-wrapper.R

#' Fit a BayesR Mixture Model (MCMC or stochastic EM)
#'
#' BayesR (Erbe et al., 2012) places a four-component mixture prior on
#' allele effects: \emph{zero}, \emph{small}, \emph{medium}, and
#' \emph{large} effect classes with proportions \code{pi_vec} and class
#' variances proportional to \code{sigma2_ah}. Use \code{method = "mcmc"}
#' for full Bayesian posterior inference, or \code{method = "em"} for
#' fast stochastic-EM point estimates (Gaussian only).
#'
#' @details
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
#' @param w Numeric design matrix (\code{n x p}). Typically the
#'   \code{$W_ah} element returned by \code{\link{construct_wah_matrix}}.
#' @param y Phenotype vector (length \code{n}). For binary traits use 0/1
#'   coding.
#' @param wtw_diag Pre-computed \code{colSums(w^2)}, length \code{p}.
#' @param X Optional fixed-effects design matrix (\code{n x q}). When
#'   supplied, the model is \eqn{y = X\alpha + W\beta + \mu + \epsilon}
#'   with a flat prior on \eqn{\alpha}. Do \strong{not} include a column
#'   of ones for the intercept; \eqn{\mu} is sampled separately.
#' @param pi_vec Initial mixture proportions for the four classes (zero,
#'   small, medium, large). Must sum to 1. Default
#'   \code{c(0.95, 0.02, 0.02, 0.01)}.
#' @param sigma2_e_init Initial residual variance. Typically
#'   \code{var(y) * 0.5}; for binary traits fix at \code{1.0}.
#' @param sigma2_ah Prior total genetic variance (required). Typically
#'   \code{var(y) * 0.5}; for binary traits use \code{1.0}.
#' @param sigma2_vec Optional explicit variance vector for EM; if
#'   \code{NULL} it is derived from \code{sigma2_ah}, \code{pi_vec}, and
#'   \code{prior_params$variance_class}.
#' @param prior_params Optional named list overriding defaults: \code{a0_e}
#'   (10), \code{a0_g} (10), \code{variance_class}
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
#'   disk as an RDS file. Set \code{FALSE} for CV loops.
#' @param save_path Optional explicit RDS path. If \code{NULL} (default),
#'   defaults to \code{"results_bayesr.Rds"} in the current working
#'   directory.
#' @param verbose If \code{TRUE}, the Rust engine streams per-iteration
#'   progress (start banner, Iter X/Y diagnostics, ESS / Geweke,
#'   completion) to stderr. Default \code{FALSE} keeps the Rust side
#'   silent. The brief R post-fit summary (model, runtime, \eqn{h^2},
#'   ESS, Geweke) prints regardless.
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
#'     values.}
#'   \item{\code{h2, sigma2_g, sigma2_e}}{Heritability and variance
#'     components.}
#'   \item{\code{runtime}}{Elapsed seconds.}
#'   \item{\code{training_metrics}}{\code{R2, RMSE, accuracy} (or
#'     \code{AUC} for binary), \code{bias}.}
#'   \item{\code{diagnostics}}{ESS and Geweke Z for key parameters
#'     (MCMC only).}
#'   \item{\code{variance_components}}{Posterior mean and 95\% CI for
#'     each class plus the mixture proportions \code{pi}.}
#'   \item{\code{rds_path}}{Path of the saved RDS file, or \code{NULL}
#'     when \code{save_rds = FALSE}.}
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(42)
#' n     <- 200
#' mcmc  <- list(n_iter = 2000L, n_burn = 1000L, n_thin = 5L, seed = 123L)
#' X_cov <- cbind(sex = rbinom(n, 1, 0.5), batch = rnorm(n))  # optional
#'
#' # ---- (A) SNP path ------------------------------------------------------
#' n_snp <- 100
#' X     <- matrix(rbinom(n * n_snp, 2, prob = runif(n_snp, 0.1, 0.5)),
#'                 n, n_snp)
#' W_snp <- construct_snp_matrix(X)$W
#' y     <- W_snp[, 1:5] %*% rnorm(5, 0, 0.5) + rnorm(n, 0, 1)
#'
#' fit_snp <- run_bayesr(
#'   w             = W_snp,
#'   y             = y,
#'   wtw_diag      = colSums(W_snp^2),
#'   X             = X_cov,           # optional
#'   sigma2_e_init = var(y) * 0.5,
#'   sigma2_ah     = var(y) * 0.5,
#'   mcmc_params   = mcmc
#' )
#' summary(fit_snp)
#'
#' # ---- (B) Microhaplotype path ------------------------------------------
#' n_block <- 50
#' hap <- matrix(sample.int(3, n * n_block * 2, replace = TRUE), nrow = n)
#' colnames(hap) <- paste0("hap_", rep(seq_len(n_block), each = 2))
#' freq <- data.frame(
#'   haplotype = paste0("hap_", rep(seq_len(n_block), each = 3)),
#'   allele    = rep(1:3, n_block),
#'   freq      = rep(c(0.5, 0.3, 0.2), n_block)
#' )
#' W_mh <- construct_wah_matrix(hap, colnames(hap), freq)$W_ah
#' y_mh <- W_mh[, 1:5] %*% rnorm(5, 0, 0.5) + rnorm(n, 0, 1)
#'
#' fit_mh <- run_bayesr(
#'   w             = W_mh,
#'   y             = y_mh,
#'   wtw_diag      = colSums(W_mh^2),
#'   X             = X_cov,           # optional
#'   sigma2_e_init = var(y_mh) * 0.5,
#'   sigma2_ah     = var(y_mh) * 0.5,
#'   mcmc_params   = mcmc
#' )
#' summary(fit_mh)
#'
#' # ---- Train/test split (scheme-agnostic) -------------------------------
#' idx  <- sample(n, 0.8 * n)
#' W_tr <- W_snp[idx, ]
#' y_tr <- y[idx]
#' fit  <- run_bayesr(
#'   w             = W_tr,
#'   y             = y_tr,
#'   wtw_diag      = colSums(W_tr^2),
#'   sigma2_e_init = var(y_tr) * 0.5,
#'   sigma2_ah     = var(y_tr) * 0.5,
#'   mcmc_params   = mcmc
#' )
#' pred <- predict(fit, W_snp[-idx, ], y[-idx])
#' pred$metrics$accuracy
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
#' Meuwissen, T. H. E., Hayes, B. J., \& Goddard, M. E. (2001).
#' Prediction of total genetic value using genome-wide dense marker
#' maps. \emph{Genetics}, 157(4), 1819-1829.
#' \doi{10.1093/genetics/157.4.1819}
#'
#' @export
run_bayesr <- function(w, y, wtw_diag,
                       X             = NULL,
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
                       save_rds      = TRUE,
                       save_path     = NULL,
                       verbose       = FALSE) {

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
  is_binary     <- response_type == "binary"

  if (is_binary && method == "em") {
    stop("response_type = 'binary' is only supported for method = 'mcmc'")
  }
  if (is.null(sigma2_ah)) stop("sigma2_ah required for both MCMC and EM")

  if (method == "mcmc") {
    mcmc_params <- modifyList(
      list(n_iter = 40000L, n_burn = 20000L, n_thin = 10L, seed = 123L),
      mcmc_params %||% list()
    )
    prior_params <- modifyList(
      list(a0_e = 10, a0_g = 10, variance_class = c(0, 0.001, 0.01, 0.1)),
      prior_params %||% list()
    )
    prior_params$b0_e <- sigma2_e_init * (prior_params$a0_e - 1)
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
    variance_class <- prior_params$variance_class %||% c(0, 0.001, 0.01, 0.1)
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
    call          = call
  )

  fit$rds_path <- maybe_save_rds(fit, save_rds, save_path)

  print_run_summary(fit)

  fit
}
