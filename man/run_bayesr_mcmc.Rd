\name{run_bayesr_mcmc}
\alias{run_bayesr_mcmc}
\title{BayesR with MCMC sampling for multiallelic-based genomic prediction (Rust Backend)}
\description{
BayesR for multiallelic-based genomic prediction with mixture prior.
}
\usage{
run_bayesr_mcmc(w, y, wtw_diag, wty, pi_vec, sigma2_vec, 
                sigma2_e_init, prior_params, mcmc_params)
}
\arguments{
  \item{w}{Training genotype matrix (n x p)}
  \item{y}{Phenotype vector (n)}
  \item{wtw_diag}{Diagonal of W'W (p)}
  \item{wty}{W'y vector (p)}
  \item{pi_vec}{Mixture proportions (4)}
  \item{sigma2_vec}{Variance components (4)}
  \item{sigma2_e_init}{Initial residual variance}
  \item{prior_params}{List of prior hyperparameters}
  \item{mcmc_params}{List of MCMC parameters (n_iter, n_burn, n_thin, seed)}
}
\value{
List containing:
  \item{beta_samples}{Posterior samples of marker effects}
  \item{gamma_samples}{Posterior samples of component assignments}
  \item{sigma2_e_samples}{Posterior samples of residual variance}
  \item{sigma2_small_samples}{Posterior samples of small effect variance}
  \item{sigma2_medium_samples}{Posterior samples of medium effect variance}
  \item{sigma2_large_samples}{Posterior samples of large effect variance}
  \item{pi_samples}{Posterior samples of mixture proportions}
}
\examples{
# See package vignette for full examples
}