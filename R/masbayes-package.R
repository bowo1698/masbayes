#' masbayes: Bayesian Genomic Prediction for SNP and Multi-allelic Markers
#'
#' Implements BayesR (four-class mixture prior) and BayesA
#' (marker-specific variance prior) for genomic prediction using biallelic
#' SNP or multi-allelic markers. Both methods are available as full MCMC
#' or stochastic EM; continuous (Gaussian) and binary (probit) traits are
#' supported natively via a high-performance Rust backend.
#'
#' @section Statistical model:
#' The general linear model shared by BayesR and BayesA is:
#' \deqn{y = \mathbf{1}\mu + W\beta + \varepsilon,
#'   \quad \varepsilon \sim \mathcal{N}(0,\,\sigma_e^2 I)}
#' where \eqn{y} is the phenotype vector, \eqn{\mu} is the intercept,
#' \eqn{W} is the marker design matrix (from
#' \code{\link{construct_wah_matrix}} for multi-allelic or
#' \code{\link{construct_snp_matrix}} for SNP markers), \eqn{\beta} is
#' the vector of allele effects, and \eqn{\sigma_e^2} is the residual
#' variance. When fixed effects are included:
#' \deqn{y = \mathbf{1}\mu + X\alpha + W\beta + \varepsilon}
#'
#' @section General assumptions:
#' \itemize{
#'   \item Phenotypic observations are conditionally independent given the
#'     genetic and fixed effects.
#'   \item Residuals \eqn{\varepsilon \sim \mathcal{N}(0,\,\sigma_e^2 I)};
#'     for binary traits the probit model with Albert-Chib data augmentation
#'     is used (liability scale, \eqn{\sigma_e^2 = 1}).
#'   \item Allele effects \eqn{\beta_j} are a priori independent across
#'     loci; the prior distribution differs between BayesR and BayesA.
#'   \item The design matrix \eqn{W} is centred (column mean zero) to
#'     ensure identifiability of the intercept \eqn{\mu}.
#'   \item For multi-allelic markers the most-frequent allele per block is
#'     treated as the baseline reference and excluded from \eqn{W}.
#' }
#'
#' @section How the model works (marker level):
#' masbayes estimates allele effects \emph{individually} at the marker
#' level, rather than aggregating them into a population-level relationship
#' matrix.
#'
#' \strong{BayesR} places a four-component mixture prior on each allele
#' effect (Erbe et al., 2012):
#' \deqn{\beta_j \mid \pi,\,\sigma_\beta^2 \;\sim\;
#'   \pi_0\,\mathcal{N}(0,0)
#'   + \pi_1\,\mathcal{N}(0,\,0.001\sigma_\beta^2)
#'   + \pi_2\,\mathcal{N}(0,\,0.01\sigma_\beta^2)
#'   + \pi_3\,\mathcal{N}(0,\,0.1\sigma_\beta^2)}
#' Most alleles are shrunk to zero (\eqn{\pi_0}), while a small proportion
#' carry small, medium, or large effects — enabling implicit variable
#' selection.
#'
#' \strong{BayesA} assigns each allele a marker-specific variance from a
#' scaled inverse chi-squared distribution (Meuwissen et al., 2001):
#' \deqn{\beta_j \mid \sigma_{\beta_j}^2 \sim \mathcal{N}(0,\,\sigma_{\beta_j}^2),
#'   \quad \sigma_{\beta_j}^2 \sim \chi^{-2}(\nu,\,S^2)}
#' allowing heavier-tailed effect-size distributions than ridge regression.
#'
#' Posterior parameters are inferred via MCMC (Gibbs sampling) or
#' stochastic EM. Genomic estimated breeding values (GEBVs) are computed
#' by summing posterior mean allele effects across all loci:
#' \deqn{\widehat{g}_i = \hat{\mu} + \mathbf{w}_i^\top \hat{\boldsymbol{\beta}}}
#' Heritability is estimated as
#' \eqn{h^2 = \mathrm{Var}(W\hat{\beta}) /
#'   [\mathrm{Var}(W\hat{\beta}) + \hat{\sigma}_e^2]}.
#'
#' @section Main functions:
#' \describe{
#'   \item{\code{\link{construct_wah_matrix}}}{Build the multi-allelic
#'     \eqn{W_{\alpha h}} design matrix from phased haplotype data.}
#'   \item{\code{\link{construct_snp_matrix}}}{Build the SNP design matrix
#'     (VanRaden centering or z-score standardisation).}
#'   \item{\code{\link{run_bayesr}}}{Fit BayesR (four-class mixture).}
#'   \item{\code{\link{run_bayesa}}}{Fit BayesA (marker-specific variance).}
#'   \item{\code{\link{predict.masbayes_bayesr}},
#'     \code{\link{predict.masbayes_bayesa}}}{Predict GEBVs and compute
#'     accuracy metrics (S3 methods).}
#'   \item{\code{\link{summary.masbayes_bayesr}},
#'     \code{\link{summary.masbayes_bayesa}}}{Full posterior report
#'     including \eqn{h^2}, ESS, Geweke, and variance components (S3
#'     methods).}
#' }
#'
#' @section References:
#' \itemize{
#'   \item Meuwissen, T. H. E., Hayes, B. J., \& Goddard, M. E. (2001).
#'     Prediction of total genetic value using genome-wide dense marker
#'     maps. \emph{Genetics}, 157(4), 1819--1829.
#'     \doi{10.1093/genetics/157.4.1819}
#'   \item Erbe, M., Hayes, B. J., Matukumalli, L. K., Goswami, S.,
#'     Bowman, P. J., Reich, C. M., Mason, B. A., \& Goddard, M. E.
#'     (2012). Improving accuracy of genomic predictions within and
#'     between dairy cattle breeds with imputed high-density single
#'     nucleotide polymorphism panels. \emph{Journal of Dairy Science},
#'     95(7), 4114--4129. \doi{10.3168/jds.2011-5019}
#'   \item Da, Y. (2015). Multi-allelic haplotype model based on genetic
#'     partition for genomic prediction and variance component estimation
#'     using SNP markers. \emph{BMC Genetics}, 16(1), 144.
#'     \doi{10.1186/s12863-015-0301-1}
#' }
#'
#' @author \strong{Maintainer}: Agus Wibowo \email{aguswibowo1698@@gmail.com}
#'
#' @name masbayes-package
NULL
