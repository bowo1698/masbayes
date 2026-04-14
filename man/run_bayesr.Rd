\name{run_bayesr}
\alias{run_bayesr}
\title{BayesR for Multi-allelic Genomic Prediction}
\description{
Fits BayesR model for genomic prediction using multi-allelic or biallelic marker matrices.
Performs automatic variable selection via a 4-component mixture prior assigning markers
to zero, small, medium, or large effect categories. Supports both MCMC for full posterior
inference and EM for fast point estimation. Accepts both W_αh (Da 2015) and G_snp
(VanRaden 2008) matrices as input.
}
\usage{
run_bayesr(
  w,
  y,
  wtw_diag,
  wty,
  pi_vec = c(0.95, 0.02, 0.02, 0.01),
  sigma2_e_init,
  sigma2_ah = NULL,
  sigma2_vec = NULL,
  prior_params = NULL,
  mcmc_params = NULL,
  em_params = NULL,
  method = c("mcmc", "em"),
  response_type = c("gaussian", "binary"),
  fold_id = 0L
)
}
\arguments{
  \item{w}{Numeric matrix (n × p). Design matrix from \code{\link{construct_wah_matrix}}
    or VanRaden-coded SNP matrix.}

  \item{y}{Numeric vector (n). Phenotype values. For \code{response_type = "binary"},
    must be 0/1.}

  \item{wtw_diag}{Numeric vector (p). Diagonal of W'W. Precompute as \code{colSums(W^2)}.}

  \item{wty}{Numeric vector (p). Cross-product W'y. Precompute as
    \code{as.vector(crossprod(W, y))}.}

  \item{pi_vec}{Numeric vector (4). Initial mixture proportions for
    \code{[zero, small, medium, large]} components. Must sum to 1.
    Default: \code{c(0.95, 0.02, 0.02, 0.01)}.}

  \item{sigma2_e_init}{Numeric. Initial residual variance. Use \code{var(y) * (1 - h2)}.
    Automatically fixed to 1.0 for \code{response_type = "binary"}.}

  \item{sigma2_ah}{Numeric. Total genetic variance for marker initialization.
    Used to compute variance class scaling: \code{varg = sigma2_ah / ((1 - pi_vec[1]) * p)}.
    Required for \code{method = "mcmc"}.}

  \item{sigma2_vec}{Numeric vector (4). Explicit variance components for EM only.
    If NULL, computed automatically from \code{sigma2_ah} and \code{variance_class}.}

  \item{prior_params}{List. Prior hyperparameters (required for \code{method = "mcmc"}):
    \describe{
      \item{a0_e}{Shape for residual variance prior (default: 10).}
      \item{a0_g}{Shape for genetic variance prior (default: 10).}
      \item{variance_class}{Scaling factors for \code{[zero, small, medium, large]}
        variance components (default: \code{c(0, 0.01, 0.1, 1)}).}
    }}

  \item{mcmc_params}{List. MCMC settings (required for \code{method = "mcmc"}):
    \describe{
      \item{n_iter}{Total iterations (integer).}
      \item{n_burn}{Burn-in (integer).}
      \item{n_thin}{Thinning interval (integer).}
      \item{seed}{Random seed (integer).}
    }}

  \item{em_params}{List. EM settings (required for \code{method = "em"}):
    \describe{
      \item{max_iter}{Maximum iterations (default: 500).}
      \item{tol}{Convergence tolerance (default: 1e-6).}
    }}

  \item{method}{Character. \code{"mcmc"} for full Bayesian inference or \code{"em"}
    for fast point estimation.}

  \item{response_type}{Character. \code{"gaussian"} for continuous traits (default) or
    \code{"binary"} for 0/1 traits via Albert-Chib data augmentation. Binary is only
    supported for \code{method = "mcmc"}.}

  \item{fold_id}{Integer. Fold identifier for cross-validation logging (default: 0L).}
}
\details{
BayesR uses a 4-component mixture prior for marker effects:

\deqn{\beta_j | \gamma_j \sim N(0, \sigma^2_{\gamma_j})}
\deqn{\gamma_j \sim Multinomial(\pi_0, \pi_S, \pi_M, \pi_L)}

where \eqn{\gamma_j \in \{0, S, M, L\}} indicates zero, small, medium, or large
effect components. Mixture proportions \eqn{\pi} are updated each iteration via
a Dirichlet posterior, enabling data-driven variable selection.

\strong{Binary traits:} When \code{response_type = "binary"}, Albert-Chib data
augmentation is applied. A latent liability \eqn{z_i} is sampled each iteration
from a truncated normal distribution conditional on the observed binary outcome,
and \eqn{\sigma^2_e} is fixed to 1.0 for identifiability. GEBV and \code{z_hat}
are on the liability scale.

\strong{variance_class tuning:} The \code{variance_class} argument controls the
relative spacing of variance components. For standardized traits (\code{var(y) ≈ 1}),
\code{c(0, 0.01, 0.1, 1)} is recommended. Wider spacing improves component
separation for sparse genetic architectures.
}
\value{
A list containing:
  \item{beta_hat}{Numeric vector (p). Posterior mean marker effects.}
  \item{mu_hat}{Numeric scalar. Posterior mean intercept.}
  \item{sigma2_e_hat}{Numeric scalar. Posterior mean residual variance (1.0 for binary).}
  \item{sigma2_g}{Numeric scalar. Variance of training GEBVs.}
  \item{h2}{Numeric scalar. Heritability estimate.}
  \item{pred_train}{Numeric vector (n). Fitted values on training set.}
  \item{beta_samples}{Matrix (n_save × p). Posterior samples of marker effects.}
  \item{gamma_samples}{Matrix (n_save × p). Component assignments
    (0 = zero, 1 = small, 2 = medium, 3 = large).}
  \item{sigma2_e_samples}{Numeric vector (n_save). Posterior samples of residual variance.}
  \item{sigma2_small_samples}{Numeric vector (n_save). Posterior samples of small variance.}
  \item{sigma2_medium_samples}{Numeric vector (n_save). Posterior samples of medium variance.}
  \item{sigma2_large_samples}{Numeric vector (n_save). Posterior samples of large variance.}
  \item{pi_samples}{Matrix (n_save × 4). Posterior samples of mixture proportions.}
  \item{mu_samples}{Numeric vector (n_save). Posterior samples of intercept.}
  \item{z_hat}{Numeric vector (n) or NULL. Posterior mean latent liability.
    Only returned when \code{response_type = "binary"}, NULL otherwise.}
}
\examples{
\dontrun{
library(masbayes)

# Simulate data
set.seed(42)
n <- 200; p <- 100
W <- matrix(rnorm(n * p), n, p)
y <- W[, 1:5] \%*\% rnorm(5, 0, 0.5) + rnorm(n, 0, 1)

wtw <- colSums(W^2)
wty <- as.vector(crossprod(W, y))

# ── Continuous trait, MCMC ──────────────────────────────────────────────────
res <- run_bayesr(
  w = W, y = y, wtw_diag = wtw, wty = wty,
  pi_vec        = c(0.90, 0.05, 0.03, 0.02),
  sigma2_e_init = var(y) * 0.5,
  sigma2_ah     = var(y) * 0.5,
  prior_params  = list(a0_e=10, a0_g=10, variance_class=c(0, 0.01, 0.1, 1)),
  mcmc_params   = list(n_iter=2000L, n_burn=1000L, n_thin=5L, seed=123L),
  method        = "mcmc"
)

gebv <- as.vector(W \%*\% res$beta_hat) + res$mu_hat
cat("Training accuracy:", round(cor(gebv, y), 3), "\n")
cat("h2:", round(res$h2, 3), "\n")

# Posterior mixture proportions
pi_post <- colMeans(res$pi_samples)
names(pi_post) <- c("Zero", "Small", "Medium", "Large")
print(round(pi_post, 3))

# ── Binary trait, MCMC (Albert-Chib) ───────────────────────────────────────
y_bin <- as.numeric(y > median(y))

res_bin <- run_bayesr(
  w = W, y = y_bin, wtw_diag = wtw,
  wty           = as.vector(crossprod(W, y_bin)),
  pi_vec        = c(0.90, 0.05, 0.03, 0.02),
  sigma2_e_init = 1.0,
  sigma2_ah     = 1.0,
  prior_params  = list(a0_e=10, a0_g=10, variance_class=c(0, 0.01, 0.1, 1)),
  mcmc_params   = list(n_iter=2000L, n_burn=1000L, n_thin=5L, seed=123L),
  method        = "mcmc",
  response_type = "binary"
)

gebv_bin <- as.vector(W \%*\% res_bin$beta_hat) + res_bin$mu_hat
cat("AUC:", round(pROC::auc(pROC::roc(y_bin, gebv_bin, quiet=TRUE)), 3), "\n")
cat("Liability z_hat range:", round(range(res_bin$z_hat), 3), "\n")

# ── EM (fast point estimation) ──────────────────────────────────────────────
res_em <- run_bayesr(
  w = W, y = y, wtw_diag = wtw, wty = wty,
  pi_vec        = c(0.90, 0.05, 0.03, 0.02),
  sigma2_e_init = var(y) * 0.5,
  sigma2_ah     = var(y) * 0.5,
  prior_params  = list(a0_e=10, a0_g=10, variance_class=c(0, 0.01, 0.1, 1)),
  em_params     = list(max_iter=500L, tol=1e-6),
  method        = "em"
)

gebv_em <- as.vector(W \%*\% res_em$beta_hat) + res_em$mu_hat
cat("EM training accuracy:", round(cor(gebv_em, y), 3), "\n")
}
}
\references{
Erbe M, et al. (2012) Improving accuracy of genomic predictions within and between
dairy cattle breeds with imputed high-density SNP panels.
\emph{Journal of Dairy Science}, 95(7): 4114--4129.

Moser G, et al. (2015) Simultaneous discovery, estimation and prediction analysis
of complex traits using a Bayesian mixture model.
\emph{PLoS Genetics}, 11(4): e1004969.

Da Y (2015) Multi-allelic haplotype model based on genetic partition for genomic
prediction and variance component estimation using SNP markers.
\emph{BMC Genetics}, 16, 144.

Albert JH, Chib S (1993) Bayesian analysis of binary and polychotomous response data.
\emph{Journal of the American Statistical Association}, 88(422): 669--679.
}
\seealso{
\code{\link{run_bayesa}}, \code{\link{construct_wah_matrix}}
}
\author{Agus Wibowo}
\keyword{models}
\keyword{regression}