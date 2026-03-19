\name{run_bayesa}
\alias{run_bayesa}
\title{BayesA for Multi-allelic Genomic Prediction}
\description{
BayesA implements marker-specific variance modeling for multi-allelic based genomic prediction,
where each marker has its own variance component with a scaled inverse chi-squared prior.
Suitable for traits controlled by many small-to-moderate effects. Supports both MCMC
for full posterior inference and EM for fast point estimation.
}
\usage{
run_bayesa(
  w,
  y,
  wtw_diag,
  wty,
  nu,
  sigma2_g,
  sigma2_e_init,
  prior_params = NULL,
  mcmc_params = NULL,
  em_params = NULL,
  method = c("mcmc", "em"),
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
  
  \item{sigma2_g}{Numeric. Initial value for genetic variance. Can be set as \code{var(y) * 0.5}
    (assuming h² ≈ 0.5) or obtained from GBLUP variance component estimation for 
    data-driven initialization.}
  
  \item{sigma2_e_init}{Numeric. Initial value for residual variance. 
    Can be set as \code{var(y) * 0.5} (assuming h² ≈ 0.5) or obtained 
    from GBLUP variance component estimation for data-driven initialization.}
  
  \item{prior_params}{List with components (required for MCMC only):
    \describe{
      \item{a0_e}{Shape parameter for residual variance inverse-gamma prior. Default value \code{10}}
    }}
  
  \item{mcmc_params}{List with components (required for method="mcmc"):
    \describe{
      \item{n_iter}{Total MCMC iterations (integer)}
      \item{n_burn}{Burn-in period (integer)}
      \item{n_thin}{Thinning interval (integer)}
      \item{seed}{Random seed (integer)}
    }}

  \item{em_params}{List of EM parameters (required for method="em"):
  \itemize{
    \item max_iter: Maximum iterations (default 500)
    \item tol: Convergence tolerance (default 1e-6)
  }}
  
  \item{method}{Algorithm choice: "mcmc" for full Bayesian inference or "em" for fast EM}

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
A list with the following components:
  \item{beta_hat}{Numeric vector (p). Posterior mean marker effects.}
  \item{mu_hat}{Numeric scalar. Intercept estimate.}
  \item{sigma2_e_hat}{Numeric scalar. Posterior mean residual variance.}
  \item{sigma2_j_hat}{Numeric vector (p). Posterior mean marker-specific variances.}
  \item{sigma2_g}{Numeric scalar. Genetic variance (variance of GEBVs).}
  \item{h2}{Numeric scalar. Heritability estimate.}
  \item{pred_train}{Numeric vector (n). Fitted values on training set.}
  \item{beta_samples}{Numeric matrix (n_save × p for MCMC, 1 × p for EM). Posterior samples of marker effects.}
  \item{sigma2_j_samples}{Numeric matrix (n_save × p for MCMC, 1 × p for EM). Posterior samples of marker-specific variances.}
  \item{sigma2_e_samples}{Numeric vector (length n_save, length 1 for EM). Posterior samples of residual variance.}
  \item{mu_samples}{Numeric vector (length n_save). Posterior samples of intercept.}
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
  a0_e = 10
)

n_iter = as.integer(100)
n_burn = as.integer(20)
n_thin = as.integer(2)
seed = as.integer(42)
tol = 1e-6

# Run BayesA with mcmc
res_mcmc <- run_bayesa(
  w = W,
  y = y,
  wtw_diag = wtw_diag,
  wty = wty,
  nu = 4.5,
  sigma2_g = var(y) * 0.5,
  sigma2_e_init = var(y) * 0.5,
  prior_params = prior_params,
  mcmc_params = list(
    n_iter = n_iter,
    n_burn = n_burn,
    n_thin = n_thin,
    seed = seed
  ),
  method = "mcmc"
)

# Run BayesA with em
res_em <- run_bayesa(
  w = W,
  y = y,
  wtw_diag = wtw_diag,
  wty = wty,
  nu = 4.5,
  sigma2_g = var(y) * 0.5,        
  sigma2_e_init = var(y) * 0.5,
  em_params = list(
    max_iter = n_iter,
    tol = tol
  ),
  method = "em",
  fold_id = 0L
)

# Posterior inference (MCMC)
beta_hat_mcmc <- colMeans(res_mcmc$beta_samples)
gebv <- W \%*\% beta_hat_mcmc

# Point estimates (EM)
beta_hat_em <- res_em$beta_samples[1,]
gebv_em <- W \%*\% beta_hat_em

# Prediction accuracy
cor(gebv, y)

# Identify markers with large variances
sigma2_j_mean <- colMeans(res_mcmc$sigma2_j_samples)
top_markers <- order(sigma2_j_mean, decreasing = TRUE)[1:10]

# Convergence diagnostics
plot(res_mcmc$sigma2_e_samples, type = "l", main = "Residual Variance Trace")
hist(beta_hat, breaks = 50, main = "Posterior Mean Effects")
}
str(res_mcmc)
str(res_em)
}
\seealso{
\code{\link{run_bayesr}}, \code{\link{construct_wah_matrix}}
}
\author{
Agus Wibowo
}
\keyword{models}
\keyword{regression}