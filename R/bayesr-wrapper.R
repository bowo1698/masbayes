# R/bayesr-wrapper.R

#' Run BayesR MCMC
#'
#' @param w_hap Haplotype W matrix (n x p_hap), or NULL
#' @param snp_files Character vector of SNP dosage file paths per chr, or NULL
#' @param y Phenotype vector (n)
#' @param pi_vec Mixture proportions (4 components)
#' @param sigma2_e_init Initial residual variance
#' @param sigma2_ah Total genetic variance prior
#' @param prior_params List of prior hyperparameters (optional)
#' @param mcmc_params List of MCMC parameters (optional)
#' @param fold_id Fold identifier for logging
#' @export
run_bayesr <- function(w_hap = NULL,
                       w_snp = NULL,
                       y,
                       pi_vec = c(0.95, 0.02, 0.02, 0.01),
                       sigma2_e_init,
                       sigma2_ah,
                       prior_params = NULL,
                       mcmc_params = NULL,
                       fold_id = 0L) {

  # Validation of at least one active component
  if (is.null(w_hap) && is.null(w_snp)) {
    stop("At least one of w_hap or w_snp must be provided")
  }

  # Automatic model detection from arguments
  use_hap <- !is.null(w_hap)
  use_snp <- !is.null(w_snp)

  model_label <- dplyr::case_when(
    use_hap && use_snp ~ "SNP_add + Hap",
    use_hap            ~ "Hap only",
    use_snp            ~ "SNP only"
  )
  message(sprintf("[Fold %d] Model: %s", fold_id, model_label))

  # Calculate n_markers for prior scaling
  n_markers <- 0L
  if (use_hap) n_markers <- n_markers + ncol(w_hap)
  if (use_snp) n_markers <- n_markers + ncol(w_snp)

  # Default MCMC params
  mcmc_params <- modifyList(
    list(n_iter = 40000L, n_burn = 20000L, n_thin = 10L, seed = 123L),
    mcmc_params %||% list()
  )

  # Default prior params
  var_class_default <- c(0, 0.0001, 0.001, 0.01)
  prior_params <- modifyList(
    list(
      a0_e           = 10,
      b0_e           = sigma2_e_init * (10 - 1),
      a0_g           = 4,
      b0_g           = if (n_markers > 0)
                         sigma2_ah * (4 - 2) / 4 / ((1 - pi_vec[1]) * n_markers)
                       else 1e-4,
      variance_class = var_class_default
    ),
    prior_params %||% list()
  )

  run_bayesr_mcmc(
    w_hap        = w_hap,
    w_snp        = w_snp,
    y            = y,
    pi_vec       = pi_vec,
    sigma2_e_init = sigma2_e_init,
    sigma2_ah    = sigma2_ah,
    prior_params = prior_params,
    mcmc_params  = mcmc_params,
    fold_id      = as.integer(fold_id)
  )
}