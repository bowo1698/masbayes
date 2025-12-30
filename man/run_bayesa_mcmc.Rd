\name{run_bayesa_mcmc}
\alias{run_bayesa_mcmc}
\title{BayesA with MCMC Sampling for Multi-allelic Genomic Prediction}
\description{
Implements BayesA for multi-allelic genomic prediction with marker-specific variance 
using Rust-accelerated marginalized Gibbs sampling. Each marker is assigned an independent variance 
component with a scaled inverse chi-squared prior.
}
\usage{
run_bayesa_mcmc(
  w, 
  y, 
  wtw_diag, 
  wty, 
  nu, 
  s_squared, 
  sigma2_e_init, 
  prior_params, 
  mcmc_params,
  fold_id = 0L
)
}
\arguments{
  \item{w}{Numeric matrix. Design matrix (n × p) from \code{\link{construct_wah_matrix}}.}
  
  \item{y}{Numeric vector. Phenotype values (length n).}
  
  \item{wtw_diag}{Numeric vector. Diagonal of W'W (length p). Precomputed as 
    \code{colSums(W^2)} for computational efficiency.}
  
  \item{wty}{Numeric vector. Cross-product W'y (length p). Precomputed as 
    \code{crossprod(W, y)}.}
  
  \item{nu}{Numeric. Degrees of freedom for marker-specific variance prior 
    (scaled inverse chi-squared). Typical values: 4-5.}
  
  \item{s_squared}{Numeric. Prior scale for marker variances. Often set as 
    \code{var(y) / p} to distribute genetic variance across markers.}
  
  \item{sigma2_e_init}{Numeric. Initial value for residual variance. 
    Can be set as \code{var(y) * 0.5} (assuming h² ≈ 0.5) or obtained 
    from GBLUP variance component estimation for data-driven initialization.}
  
  \item{prior_params}{List with components:
    \describe{
      \item{a0_e}{Shape parameter for residual variance inverse-gamma prior}
      \item{b0_e}{Scale parameter for residual variance inverse-gamma prior}
    }}
  
  \item{mcmc_params}{List with components:
    \describe{
      \item{n_iter}{Total MCMC iterations (integer)}
      \item{n_burn}{Burn-in period (integer)}
      \item{n_thin}{Thinning interval (integer)}
      \item{seed}{Random seed (integer)}
    }}
  
  \item{fold_id}{Integer. Fold identifier for cross-validation (default: 0L).}
}
\details{
BayesA assumes each marker has its own variance component with a scaled 
inverse chi-squared prior:

\deqn{\beta_j | \sigma^2_j \sim N(0, \sigma^2_j)}
\deqn{\sigma^2_j \sim \nu S^2 / \chi^2_\nu}

This allows markers to have heterogeneous effects, making BayesA suitable for 
traits controlled by many small-to-moderate effects. Unlike BayesR, BayesA does 
not perform variable selection - all markers contribute to prediction.

\strong{Prior Specification:}

The hyperparameter \eqn{\nu} controls the prior belief about marker variance 
heterogeneity. Smaller values (e.g., 4) allow more variance heterogeneity, 
while larger values shrink marker variances toward a common value.

\strong{Computational Notes:}

The Rust implementation uses marginalized Gibbs sampling for efficient MCMC 
convergence. Sufficient statistics (\code{wtw_diag}, \code{wty}) are precomputed 
in R to minimize overhead in the sampling loop.
}
\value{
A list with three components:
  \item{beta_samples}{Numeric matrix (n_save × p). Posterior samples of marker 
    effect sizes, where n_save = (n_iter - n_burn) / n_thin.}
  
  \item{sigma2_j_samples}{Numeric matrix (n_save × p). Posterior samples of 
    marker-specific variances.}
  
  \item{sigma2_e_samples}{Numeric vector (length n_save). Posterior samples of 
    residual variance.}
}
\references{
Meuwissen THE, Hayes BJ, Goddard ME (2001). Prediction of total genetic value 
  using genome-wide dense marker maps. \emph{Genetics}, 157(4): 1819-1829.

Zhou X, Carbonetto P, Stephens M (2013). Polygenic modeling with Bayesian sparse 
  linear mixed models. \emph{PLOS Genetics}, 9(2): e1003264.
}
\examples{
\dontrun{
# See ?construct_wah_matrix for data preparation

# Assume W and y are already prepared
W <- train_Wah$W_ah
y <- rnorm(nrow(W))

# Precompute sufficient statistics
wtw_diag <- as.numeric(colSums(W^2))
wty <- as.vector(crossprod(W, y))

# Prior parameters
prior_params <- list(
  a0_e = 3.0,
  b0_e = var(y) * 0.5 * 2
)

# MCMC parameters
mcmc_params <- list(
  n_iter = as.integer(100),
  n_burn = as.integer(20),
  n_thin = as.integer(2),
  seed = as.integer(42)
)

# Run BayesA
res <- run_bayesa_mcmc(
  w = W,
  y = y,
  wtw_diag = wtw_diag,
  wty = wty,
  nu = 4.5,
  s_squared = var(y) / ncol(W),
  sigma2_e_init = var(y) * 0.5,
  prior_params = prior_params,
  mcmc_params = mcmc_params
)

# Posterior inference
beta_hat <- colMeans(res$beta_samples)
gebv <- W \%*\% beta_hat

# Prediction accuracy
cor(gebv, y)

# Identify markers with large variances
sigma2_j_mean <- colMeans(res$sigma2_j_samples)
top_markers <- order(sigma2_j_mean, decreasing = TRUE)[1:10]

# Convergence diagnostics
plot(res$sigma2_e_samples, type = "l", main = "Residual Variance Trace")
hist(beta_hat, breaks = 50, main = "Posterior Mean Effects")
}
}
\seealso{
\code{\link{run_bayesr_mcmc}}, \code{\link{construct_wah_matrix}}
}
\author{
Agus Wibowo
}
\keyword{models}
\keyword{regression}