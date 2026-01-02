\name{masbayes-package}
\alias{masbayes-package}
\alias{masbayes}
\docType{package}
\title{Rust-Accelerated MCMC Backend for Multi-Allelic Genomic Prediction}
\description{
MasBayes supports multi-allelic-based markers for genomic prediction, where markers such as haplotypes
or microhaplotypes can be used as predictors directly feeding into prediction models without being
decomposed into biallelic markers. We implemented the W_αh matrix as described by Da, Y. (2015)
and developed BayesA and BayesR models specifically for multiallelic markers. Both matrix constructions
and Bayesian models were built on Rust programming to optimise computational efficiency rather than
purely using the R implementation. In addition, we also implemented marginalised Gibbs sampling
for Bayesian models to reduce correlation between parameters within the MCMC chain and hasten
convergence, while baseline allele dropping was also implemented to only estimate informative
alleles per haplotype block.
}
\details{
\strong{Package Overview}

The \pkg{masbayes} package implements Bayesian variable selection methods for genomic 
prediction using Rust backend. The package is specifically designed 
for multi-allelic microhaplotype data.

\strong{Main Functions}

\itemize{
  \item \code{\link{construct_wah_matrix}}: W_αh matrix construction for multi-allelic markers
  \itemize{
    \item Allele frequency-based coding scheme
    \item Optional baseline allele dropping to avoid collinearity
    \item Training/test set consistency via reference structure
    \item Handles microhaplotypes and phased SNP blocks
  }

  \item \code{\link{run_bayesr}}: BayesR with 4-component mixture prior
  \itemize{
    \item Zero effect component (markers with no effect)
    \item Small, medium, and large effect variance components
    \item Automatic component assignment via MCMC
    \item Mixture proportions estimated from data
  }
  
  \item \code{\link{run_bayesa
}}: BayesA with marker-specific variance
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

1. Prepare genotype matrix (Wah) and phenotypes (y)

2. Calculate summary statistics: WtW diagonal and Wty

3. Run MCMC sampling via \code{run_bayesr()} or \code{run_bayesa()}

4. Extract posterior means for genomic prediction

5. Calculate breeding values: GEBV = W \%*\% beta_hat

\strong{Statistical Background}

BayesR implements a mixture model:
\deqn{\beta_j | \gamma_j \sim N(0, \sigma^2_{\gamma_j})}
where \eqn{\gamma_j \in \{0, 1, 2, 3\}} indicates component membership.

BayesA uses marker-specific variances:
\deqn{\beta_j \sim N(0, \sigma^2_j)}
\deqn{\sigma^2_j \sim \chi^{-2}(\nu, S^2)}

Both methods use marginalised Gibbs sampling for posterior inference.

\strong{System Requirements}

\itemize{
  \item R >= 4.0.0
  \item Rust >= 1.85.0 (required for installation from source)
  \item For installation: \code{devtools::install_github("bowo1698/masbayes")}
}
}
\author{
Agus Wibowo \email{aguswibowo1698@gmail.com}

Maintainer: Agus Wibowo \email{aguswibowo1698@gmail.com}
}
\references{
Erbe M, et al. (2012) Improving accuracy of genomic predictions within and between 
dairy cattle breeds with imputed high-density single nucleotide polymorphism panels. 
\emph{Journal of Dairy Science}, 95(7), 4114-4129.

Meuwissen TH, Hayes BJ, Goddard ME (2001) Prediction of total genetic value using 
genome-wide dense marker maps. \emph{Genetics}, 157(4), 1819-1829.

Moser G, et al. (2015) Simultaneous discovery, estimation and prediction analysis 
of complex traits using a Bayesian mixture model. \emph{PLoS Genetics}, 11(4), e1004969.

Da, Y. (2015). Multi-allelic haplotype model based on genetic partition for genomic
prediction and variance component estimation using SNP markers. \emph{BMC Genetics},
16(144).
}
\seealso{
\code{\link{run_bayesr}}, \code{\link{run_bayesa}}

Useful links:
\itemize{
  \item GitHub repository: \url{https://github.com/bowo1698/masbayes}
  \item Report bugs: \url{https://github.com/bowo1698/masbayes/issues}
}
}
\examples{
# See function documentation for detailed examples
?construct_wah_matrix
?run_bayesr
?run_bayesa
}
\keyword{package}
\keyword{models}
\keyword{regression}