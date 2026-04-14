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
#' @param prior_params Prior hyperparameters (for MCMC)
#' @param mcmc_params MCMC parameters (for method="mcmc")
#' @param em_params EM parameters (for method="em")
#' @param method Either "mcmc" or "em"
#' @param fold_id Fold identifier
#' @export
run_bayesr <- function(w, y, wtw_diag, wty, 
                       pi_vec = c(0.95, 0.02, 0.02, 0.01),
                       sigma2_e_init,
                       sigma2_ah = NULL,
                       sigma2_vec = NULL,
                       prior_params = NULL,
                       mcmc_params = NULL,
                       em_params = NULL,
                       method = c("mcmc", "em"),
                       response_type = c("gaussian", "binary"),
                       fold_id = 0L) {
  
  method <- match.arg(method)
  response_type <- match.arg(response_type)
  is_binary     <- response_type == "binary"

  if (is_binary && method == "em") {
    stop("response_type = 'binary' is only supported for method = 'mcmc'")
  }
  
  if (method == "mcmc") {
    if (is.null(sigma2_ah)) stop("sigma2_ah required for MCMC")
    
    # Default MCMC params
    mcmc_params <- modifyList(
      list(n_iter = 40000L, n_burn = 20000L, n_thin = 10L, seed = 123L),
      mcmc_params %||% list()
    )
    # Default prior params
    prior_params <- modifyList(
      list(
        a0_e = 10,
        a0_g = 10,
        variance_class = c(0, 0.001, 0.01, 0.1)
      ),
      prior_params %||% list()
    )

    # Hitung derived params dari high-level inputs
    prior_params$b0_e <- sigma2_e_init * (prior_params$a0_e - 1)
    prior_params$b0_g <- sigma2_ah * (prior_params$a0_g - 2) / 
                        prior_params$a0_g / (1 - pi_vec[1])
    run_bayesr_mcmc(w, y, wtw_diag, wty, pi_vec, 
                    sigma2_e_init, sigma2_ah, prior_params, 
                    mcmc_params, fold_id, is_binary)
  } else {
    if (is.null(sigma2_ah)) stop("sigma2_ah required for EM")
    
    em_params <- modifyList(
      list(max_iter = 500L, tol = 1e-6),
      em_params %||% list()
    )
    
    # Hitung sigma2_vec konsisten dengan MCMC
    variance_class <- if (!is.null(prior_params$variance_class)) {
      prior_params$variance_class
    } else {
      c(0, 0.001, 0.01, 0.1)
    }
    varg_init     <- as.numeric(sigma2_ah / ((1 - pi_vec[1]) * ncol(w)))
    sigma2_vec    <- as.vector(variance_class) * varg_init
    sigma2_vec[1] <- 0

    run_bayesr_em(w, y, wtw_diag, wty, pi_vec, sigma2_vec,
                  sigma2_e_init, em_params, fold_id)
  }
}