\name{run_bayesr_mcmc}
\alias{run_bayesr_mcmc}
\title{BayesR with MCMC Sampling for Multi-allelic Genomic Prediction}
\description{
Implements BayesR for multi-allelic genomic prediction with a 4-component mixture 
prior using Rust-accelerated marginalized Gibbs sampling. Performs automatic variable selection 
by assigning markers to zero, small, medium, or large effect size categories.
}
\usage{
run_bayesr_mcmc(
  w, 
  y, 
  wtw_diag, 
  wty, 
  pi_vec, 
  sigma2_vec, 
  sigma2_e_init, 
  sigma2_ah,
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
  
  \item{pi_vec}{Numeric vector of length 4. Initial mixture proportions for 
    [zero, small, medium, large] effect components. Must sum to 1. 
    Typical: \code{c(0.85, 0.08, 0.05, 0.02)}.}
  
  \item{sigma2_vec}{Numeric vector of length 4. Variance components for 
    [zero, small, medium, large] effects. First element should be near-zero 
    (e.g., 1e-8) rather than exact 0 to avoid numerical issues. 
    Typical: \code{c(1e-8, 0.001, 0.01, 0.1)}.}
  
  \item{sigma2_e_init}{Numeric. Initial value for residual variance. 
    Can be set as \code{var(y) * 0.5} (assuming h² ≈ 0.5) or obtained 
    from GBLUP variance component estimation for data-driven initialization.}
  
  \item{sigma2_ah}{Numeric. Initial total genetic variance for marker effect 
    initialization. Suggested: \code{var(y) * 0.5} or genetic variance from GBLUP.}
  
  \item{prior_params}{List with components:
    \describe{
      \item{a0_e, b0_e}{Shape and scale for residual variance inverse-gamma prior}
      \item{a0_small, b0_small}{Shape and scale for small effect variance prior}
      \item{a0_medium, b0_medium}{Shape and scale for medium effect variance prior}
      \item{a0_large, b0_large}{Shape and scale for large effect variance prior}
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
BayesR uses a 4-component mixture prior for marker effects:

\deqn{\beta_j | \gamma_j \sim N(0, \sigma^2_{\gamma_j})}
\deqn{\gamma_j \sim Multinomial(\pi_0, \pi_S, \pi_M, \pi_L)}

where \eqn{\gamma_j \in \{0, S, M, L\}} indicates zero, small, medium, or large 
effect components. The mixture proportions \eqn{\pi} are estimated from data using 
a Dirichlet prior.

\strong{Variable Selection:}

Markers assigned to the zero component (\eqn{\gamma_j = 0}) have negligible 
effects, effectively performing Bayesian variable selection. This makes BayesR 
particularly suitable for traits controlled by a mixture of effect sizes.

\strong{Prior Specification:}

The variance components \code{sigma2_vec} can be initialized as multiples of 
genetic variance divided by total markers. The mixture proportions \code{pi_vec} 
are typically set with most markers in the zero component (e.g., 85\%), reflecting 
the assumption that most markers have small or no effects.

\strong{Computational Notes:}

The Rust implementation uses marginalized Gibbs sampling with log-sum-exp trick 
for numerical stability. Component assignments are sampled jointly with effect 
sizes for efficient mixing.
}
\value{
A list with seven components:
  \item{beta_samples}{Numeric matrix (n_save × p). Posterior samples of marker 
    effect sizes.}
  
  \item{gamma_samples}{Numeric matrix (n_save × p). Posterior samples of component 
    assignments (0 = zero, 1 = small, 2 = medium, 3 = large).}
  
  \item{sigma2_e_samples}{Numeric vector (length n_save). Posterior samples of 
    residual variance.}
  
  \item{sigma2_small_samples}{Numeric vector (length n_save). Posterior samples 
    of small effect variance component.}
  
  \item{sigma2_medium_samples}{Numeric vector (length n_save). Posterior samples 
    of medium effect variance component.}
  
  \item{sigma2_large_samples}{Numeric vector (length n_save). Posterior samples 
    of large effect variance component.}
  
  \item{pi_samples}{Numeric matrix (n_save × 4). Posterior samples of mixture 
    proportions.}
}
\references{
Erbe M, Hayes BJ, Matukumalli LK, et al. (2012). Improving accuracy of genomic 
  predictions within and between dairy cattle breeds with imputed high-density 
  single nucleotide polymorphism panels. \emph{Journal of Dairy Science}, 
  95(7): 4114-4129.

Moser G, Lee SH, Hayes BJ, et al. (2015). Simultaneous discovery, estimation and 
  prediction analysis of complex traits using a bayesian mixture model. 
  \emph{PLOS Genetics}, 11(4): e1004969.

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
  b0_e = 0.5,
  a0_small = 3.0,
  b0_small = 0.0005,
  a0_medium = 3.0,
  b0_medium = 0.005,
  a0_large = 3.0,
  b0_large = 0.05
)

# MCMC parameters
mcmc_params <- list(
  n_iter = 10000L,
  n_burn = 2000L,
  n_thin = 5L,
  seed = 123L
)

# Run BayesR
res <- run_bayesr_mcmc(
  w = W,
  y = y,
  wtw_diag = wtw_diag,
  wty = wty,
  pi_vec = c(0.85, 0.08, 0.05, 0.02),
  sigma2_vec = c(1e-8, 0.001, 0.01, 0.1),
  sigma2_e_init = var(y) * 0.5,
  sigma2_ah = var(y) * 0.5,
  prior_params = prior_params,
  mcmc_params = mcmc_params,
  fold_id = 0L
)

# Posterior inference
beta_hat <- colMeans(res$beta_samples)
gebv <- W \%*\% beta_hat

# Prediction accuracy
cor(gebv, y)

# Posterior mixture proportions
pi_post <- colMeans(res$pi_samples)
names(pi_post) <- c("Zero", "Small", "Medium", "Large")
print(pi_post)

# Identify markers with non-zero effects
gamma_mode <- apply(res$gamma_samples, 2, function(x) {
  as.numeric(names(sort(table(x), decreasing = TRUE)[1]))
})
important_markers <- which(gamma_mode > 0)

# Convergence diagnostics
plot(res$sigma2_e_samples, type = "l", main = "Residual Variance Trace")
matplot(res$pi_samples, type = "l", main = "Mixture Proportions", 
        ylab = "Proportion", col = 1:4, lty = 1)
legend("topright", legend = c("Zero", "Small", "Medium", "Large"), 
       col = 1:4, lty = 1)
}
}
\seealso{
\code{\link{run_bayesa_mcmc}}, \code{\link{construct_wah_matrix}}
}
\author{
Agus Wibowo
}
\keyword{models}
\keyword{regression}