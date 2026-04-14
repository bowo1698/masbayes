# R/bayesa-wrapper.R

#' Run BayesA with choice of algorithm
#' 
#' @param w Design matrix
#' @param y Phenotype vector
#' @param wtw_diag Diagonal of W'W
#' @param wty W'y vector
#' @param nu Degrees of freedom
#' @param sigma2_g Initial genetic variance
#' @param sigma2_e_init Initial residual variance
#' @param prior_params Prior hyperparameters (for MCMC)
#' @param mcmc_params MCMC parameters (for method="mcmc")
#' @param em_params EM parameters (for method="em")
#' @param method Either "mcmc" or "em"
#' @param fold_id Fold identifier
#' @export
run_bayesa <- function(w, y, wtw_diag, wty,
                       nu = 4.5,
                       sigma2_g, sigma2_e_init,
                       prior_params = NULL,
                       mcmc_params = NULL,
                       em_params = NULL,
                       method = c("mcmc", "em"),
                       response_type = c("gaussian", "binary"),
                       fold_id = 0L) {
  
  method        <- match.arg(method)
  response_type <- match.arg(response_type)
  is_binary     <- response_type == "binary"

  if (is_binary && method == "em") {
    stop("response_type = 'binary' is only supported for method = 'mcmc'")
  }
  
  sum_2pq   <- sum(apply(w, 2, var))
  s_squared <- sigma2_g * (nu - 2) / (nu * sum_2pq)
  
  if (method == "mcmc") {
    mcmc_params <- modifyList(
      list(n_iter = 40000L, n_burn = 20000L, n_thin = 10L, seed = 123L),
      mcmc_params %||% list()
    )
    # Default prior params
    prior_params <- modifyList(
      list(a0_e = 10),
      prior_params %||% list()
    )
    prior_params$b0_e <- sigma2_e_init * (prior_params$a0_e - 1)
    run_bayesa_mcmc(w, y, wtw_diag, wty, nu, s_squared, 
                    sigma2_e_init, prior_params, mcmc_params, fold_id, is_binary)
  } else {
    em_params <- modifyList(
      list(max_iter = 500L, tol = 1e-6),
      em_params %||% list()
    )
    run_bayesa_em(w, y, wtw_diag, wty, nu, s_squared,
                  sigma2_e_init, em_params, fold_id)
  }
}