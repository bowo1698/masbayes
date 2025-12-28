\name{masbayes-package}
\alias{masbayes-package}
\alias{masbayes}
\docType{package}
\title{Rust-Accelerated MCMC Backend for Multi-Allelic Genomic Prediction}
\description{
High-performance MCMC sampling engines for BayesR and BayesA genomic prediction methods 
with native support for multi-allelic microhaplotype markers. Provides approximately 10x 
speedup over pure R implementations through Rust-accelerated computation while maintaining 
identical statistical properties.
}
\details{
\strong{Package Overview}

The \pkg{masbayes} package implements Bayesian variable selection methods for genomic 
prediction using a high-performance Rust backend. The package is specifically designed 
for multi-allelic microhaplotype data but also works with standard bi-allelic SNPs.

\strong{Main Functions}

\itemize{
  \item \code{\link{run_bayesr_mcmc}}: BayesR with 4-component mixture prior
  \itemize{
    \item Zero effect component (markers with no effect)
    \item Small, medium, and large effect variance components
    \item Automatic component assignment via MCMC
    \item Mixture proportions estimated from data
  }
  
  \item \code{\link{run_bayesa_mcmc}}: BayesA with marker-specific variance
  \itemize{
    \item Independent variance for each marker
    \item Scaled inverse chi-squared prior
    \item Suitable for traits with many small effects
  }
}

\strong{Key Features}

\itemize{
  \item \strong{Multi-allelic support}: Native handling of microhaplotype blocks
  \item \strong{Scalability}: Efficient for large-scale genomic data
  \item \strong{Accuracy}: Identical statistical results to R implementations
  \item \strong{Diagnostics}: Built-in convergence monitoring (ESS, Geweke)
}

\strong{Typical Workflow}

1. Prepare genotype matrix (W) and phenotypes (y)

2. Calculate summary statistics: WtW diagonal and Wty

3. Run MCMC sampling via \code{run_bayesr_mcmc()} or \code{run_bayesa_mcmc()}

4. Extract posterior means for genomic prediction

5. Calculate breeding values: GEBV = W \%*\% beta_hat

\strong{Performance Benchmarks}

Typical speedup on standard hardware (M1 MacBook Pro, 16GB RAM):
\itemize{
  \item 1K markers, 50K iterations: 30 seconds (vs 5 min in R)
  \item 10K markers, 50K iterations: 5 minutes (vs 50 min in R)
  \item 100K markers, 50K iterations: 48 minutes (vs 8 hours in R)
}

\strong{Statistical Background}

BayesR implements a mixture model:
\deqn{\beta_j | \gamma_j \sim N(0, \sigma^2_{\gamma_j})}
where \eqn{\gamma_j \in \{0, 1, 2, 3\}} indicates component membership.

BayesA uses marker-specific variances:
\deqn{\beta_j \sim N(0, \sigma^2_j)}
\deqn{\sigma^2_j \sim \chi^{-2}(\nu, S^2)}

Both methods use Gibbs sampling for posterior inference.

\strong{System Requirements}

\itemize{
  \item R >= 4.0.0
  \item Rust >= 1.85.0 (required for installation from source)
  \item For installation: \code{devtools::install_github("bowo1698/masbayes")}
}
}
\author{
Agus Bowo Wibowo \email{aguswibowo1698@gmail.com}

Maintainer: Agus Bowo Wibowo \email{aguswibowo1698@gmail.com}
}
\references{
Erbe M, et al. (2012) Improving accuracy of genomic predictions within and between 
dairy cattle breeds with imputed high-density single nucleotide polymorphism panels. 
\emph{Journal of Dairy Science}, 95(7), 4114-4129.

Meuwissen TH, Hayes BJ, Goddard ME (2001) Prediction of total genetic value using 
genome-wide dense marker maps. \emph{Genetics}, 157(4), 1819-1829.

Moser G, et al. (2015) Simultaneous discovery, estimation and prediction analysis 
of complex traits using a Bayesian mixture model. \emph{PLoS Genetics}, 11(4), e1004969.
}
\seealso{
\code{\link{run_bayesr_mcmc}}, \code{\link{run_bayesa_mcmc}}

Useful links:
\itemize{
  \item GitHub repository: \url{https://github.com/bowo1698/masbayes}
  \item Report bugs: \url{https://github.com/bowo1698/masbayes/issues}
}
}
\examples{
# See function documentation for detailed examples
?run_bayesr_mcmc
?run_bayesa_mcmc
}
\keyword{package}
\keyword{models}
\keyword{regression}