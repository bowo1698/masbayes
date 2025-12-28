\name{run_bayesa_mcmc}
\alias{run_bayesa_mcmc}
\title{BayesA with MCMC sampling for multiallelic-based genomic prediction (Rust Backend)}
\description{
BayesA for multiallelic-based genomic prediction with marker-specific variance.
}
\usage{
run_bayesa_mcmc(w, y, wtw_diag, wty, nu, s_squared, 
                sigma2_e_init, prior_params, mcmc_params)
}
\arguments{
  \item{w}{Training genotype matrix (n x p)}
  \item{y}{Phenotype vector (n)}
  \item{wtw_diag}{Diagonal of W'W (p)}
  \item{wty}{W'y vector (p)}
  \item{nu}{Degrees of freedom for marker variance prior}
  \item{s_squared}{Prior scale for marker variances}
  \item{sigma2_e_init}{Initial residual variance}
  \item{prior_params}{List of prior hyperparameters}
  \item{mcmc_params}{List of MCMC parameters (n_iter, n_burn, n_thin, seed)}
}
\value{
List containing:
  \item{beta_samples}{Posterior samples of marker effects}
  \item{sigma2_j_samples}{Posterior samples of marker-specific variances}
  \item{sigma2_e_samples}{Posterior samples of residual variance}
}
\examples{
# See package vignette for full examples
}