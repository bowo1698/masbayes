\name{masbayes-package}
\alias{masbayes-package}
\alias{masbayes}
\docType{package}
\title{masbayes: Bayesian Genomic Prediction for Multi-Allelic Markers}
\description{
Implements BayesR and BayesA models for genomic prediction using multi-allelic markers
such as microhaplotypes and SNPs. Built on a Rust backend for computational efficiency, with native
support for the W_αh matrix coding scheme of Da (2015). Supports both continuous and
binary traits via Albert-Chib data augmentation.
}

\strong{Main Functions}

\itemize{
  \item \code{\link{construct_wah_matrix}} — construct W_αh matrix from phased haplotype data
  \item \code{\link{run_bayesr}} — BayesR with 4-component mixture prior
  \item \code{\link{run_bayesa}} — BayesA with marker-specific variance
}

\strong{Key Features}

\itemize{
  \item Multi-allelic native: microhaplotype blocks handled directly without biallelic decomposition
  \item Universal marker input: accepts both W_αh (Da 2015) and G_snp (VanRaden 2008) matrices
  \item Binary trait support: Albert-Chib data augmentation via \code{response_type = "binary"}
  \item Marginalized Gibbs sampling for faster convergence and reduced parameter correlation
  \item Built-in MCMC diagnostics: ESS and Geweke Z-score
  \item Cross-generation prediction via consistent block definitions
}

\strong{Typical Workflow}

1. Prepare genotype matrix (Wah) and phenotypes (y)

2. Calculate summary statistics: WtW diagonal and Wty

3. Run MCMC sampling via \code{run_bayesr()} or \code{run_bayesa()}

4. Extract posterior means for genomic prediction

5. Calculate breeding values: GEBV = W \%*\% beta_hat

\strong{Statistical Background}

BayesR implements a mixture model:
\deqn{\beta_j | \gamma_j \sim N(0, \sigma^2_{\gamma_j})}{\beta[j] | \gamma[j] ~ N(0, \sigma^2[\gamma[j]])}
where \eqn{\gamma_j \in \{0, 1, 2, 3\}}{\gamma[j] in {0,1,2,3}} indicates component membership.

BayesA uses marker-specific variances:
\deqn{\beta_j \sim N(0, \sigma^2_j)}{\beta[j] ~ N(0, \sigma^2[j])}
\deqn{\sigma^2_j \sim \chi^{-2}(\nu, S^2)}{\sigma^2[j] ~ \chi^{-2}(\nu, S^2)}

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