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
#' @param fold_id Fold identifier
#' @export
run_bayesa <- function(w, y, wtw_diag, wty,
                       nu = 4.5,
                       s_squared, sigma2_e_init,
                       prior_params = NULL,
                       mcmc_params = NULL,
                       fold_id = 0L) {
  
    mcmc_params <- modifyList(
      list(n_iter = 40000L, n_burn = 20000L, n_thin = 10L, seed = 123L),
      mcmc_params %||% list()
    )
    # Default prior params
    prior_params <- modifyList(
      list(a0_e = 10, b0_e = sigma2_e_init * (10 - 1)),
      prior_params %||% list()
    )
    run_bayesa_mcmc(w, y, wtw_diag, wty, nu, s_squared, 
                    sigma2_e_init, prior_params, mcmc_params, fold_id)
}