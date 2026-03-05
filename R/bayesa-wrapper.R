# R/bayesa-wrapper.R

#' Run BayesA MCMC
#'
#' @param w_hap Haplotype W matrix (n x p_hap), or NULL
#' @param snp_files Character vector of SNP dosage file paths per chr, or NULL
#' @param y Phenotype vector (n)
#' @param nu Degrees of freedom for marker variance prior
#' @param s_squared Prior scale for marker variances
#' @param sigma2_e_init Initial residual variance
#' @param prior_params List of prior hyperparameters (optional)
#' @param mcmc_params List of MCMC parameters (optional)
#' @param fold_id Fold identifier for logging
#' @export
run_bayesa <- function(w_hap = NULL,
                       w_snp = NULL, 
                       y,
                       nu = 4.5,
                       s_squared,
                       sigma2_e_init,
                       prior_params = NULL,
                       mcmc_params = NULL,
                       fold_id = 0L) {

  if (is.null(w_hap) && is.null(w_snp)) {
    stop("At least one of w_hap or snp_files must be provided")
  }

  use_hap <- !is.null(w_hap)
  use_snp <- !is.null(w_snp)

  model_label <- dplyr::case_when(
    use_hap && use_snp ~ "SNP_add + Hap",
    use_hap            ~ "Hap only",
    use_snp            ~ "SNP only"
  )
  message(sprintf("[Fold %d] BayesA Model: %s", fold_id, model_label))

  mcmc_params <- modifyList(
    list(n_iter = 40000L, n_burn = 20000L, n_thin = 10L, seed = 123L),
    mcmc_params %||% list()
  )

  prior_params <- modifyList(
    list(a0_e = 10, b0_e = sigma2_e_init * (10 - 1)),
    prior_params %||% list()
  )

  run_bayesa_mcmc(
    w_hap         = w_hap,
    w_snp         = w_snp,
    y             = y,
    nu            = nu,
    s_squared     = s_squared,
    sigma2_e_init = sigma2_e_init,
    prior_params  = prior_params,
    mcmc_params   = mcmc_params,
    fold_id       = as.integer(fold_id)
  )
}