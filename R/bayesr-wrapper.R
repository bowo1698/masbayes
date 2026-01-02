# R/bayesr-wrapper.R

#' Run BayesR with choice of algorithm
#' 
#' @param w Design matrix
#' @param y Phenotype vector
#' @param wtw_diag Diagonal of W'W
#' @param wty W'y vector
#' @param pi_vec Mixture proportions
#' @param sigma2_vec Variance components
#' @param sigma2_e_init Initial residual variance
#' @param sigma2_ah Total genetic variance (for MCMC)
#' @param allele_freqs Allele frequency for shrinkage
#' @param prior_params Prior hyperparameters (for MCMC)
#' @param mcmc_params MCMC parameters (for method="mcmc")
#' @param em_params EM parameters (for method="em")
#' @param method Either "mcmc" or "em"
#' @param fold_id Fold identifier
#' @export
run_bayesr <- function(w, y, wtw_diag, wty, 
                       pi_vec, sigma2_vec, sigma2_e_init,
                       sigma2_ah = NULL,
                       allele_freqs = NULL,
                       use_adaptive_grid = FALSE,
                       prior_params = NULL,
                       mcmc_params = NULL,
                       em_params = NULL,
                       method = c("mcmc", "em"),
                       fold_id = 0L) {
  
  method <- match.arg(method)

  if (is.null(allele_freqs)) {
    stop("allele_freqs required for frequency-based shrinkage")
  }
  
  if (method == "mcmc") {
    if (is.null(sigma2_ah) || is.null(prior_params) ||
        is.null(mcmc_params) || is.null(allele_freqs)) {
      stop("For MCMC: sigma2_ah, prior_params, and mcmc_params required")
    }
    run_bayesr_mcmc(w, y, wtw_diag, wty, pi_vec, sigma2_vec,
                    sigma2_e_init, sigma2_ah, allele_freqs,
                    use_adaptive_grid,
                    prior_params, mcmc_params, fold_id)
  } else {
    if (is.null(em_params)) {
      em_params <- list(max_iter = 500L, tol = 1e-6)
    }
    run_bayesr_em(w, y, wtw_diag, wty, pi_vec, sigma2_vec, 
                  sigma2_e_init, allele_freqs, use_adaptive_grid, em_params, fold_id)
  }
}