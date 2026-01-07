# R/bayesa-wrapper.R

#' Run BayesA with choice of algorithm
#' 
#' @param w Design matrix
#' @param y Phenotype vector
#' @param wtw_diag Diagonal of W'W
#' @param wty W'y vector
#' @param nu Degrees of freedom
#' @param s_squared Prior scale
#' @param sigma2_e_init Initial residual variance
#' @param prior_params Prior hyperparameters (for MCMC)
#' @param mcmc_params MCMC parameters (for method="mcmc")
#' @param em_params EM parameters (for method="em")
#' @param method Either "mcmc" or "em"
#' @param fold_id Fold identifier
#' @export
run_bayesa <- function(w, y, wtw_diag, wty,
                       nu, s_squared, sigma2_e_init,
                       prior_params = NULL,
                       mcmc_params = NULL,
                       em_params = NULL,
                       method = c("mcmc", "em"),
                       fold_id = 0L) {
  
  method <- match.arg(method)
  
  if (method == "mcmc") {
    if (is.null(prior_params) || is.null(mcmc_params)) {
      stop("For MCMC: prior_params and mcmc_params required")
    }
    run_bayesa_mcmc(w, y, wtw_diag, wty, nu, s_squared, 
                    sigma2_e_init, prior_params, mcmc_params, fold_id)
  } else {
    if (is.null(em_params)) {
      em_params <- list(max_iter = 500L, tol = 1e-6)
    }
    run_bayesa_em(w, y, wtw_diag, wty, nu, s_squared, 
                  sigma2_e_init, em_params, fold_id)
  }
}