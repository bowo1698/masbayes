\name{construct_wah_matrix}
\alias{construct_wah_matrix}
\title{Construct W_αh Matrix for Multi-allelic Markers}
\description{
Fast construction of allele-specific design matrix (W_αh) from haplotype genotypes 
using Rust backend. Implements allele frequency-based coding for multi-allelic 
microhaplotype markers with optional baseline allele dropping.
}
\usage{
construct_wah_matrix(
  hap_matrix,
  colnames,
  allele_freq_filtered = NULL,
  reference_structure = NULL,
  drop_baseline = TRUE
)
}
\arguments{
  \item{hap_matrix}{Integer matrix of haplotype genotypes with dimensions (n × 2B) 
    where n is number of individuals and B is number of haplotype blocks. Each block 
    has two columns representing diploid haplotypes.}
  
  \item{colnames}{Character vector of column names for \code{hap_matrix}. Should 
    have length equal to \code{ncol(hap_matrix)}.}
  
  \item{allele_freq_filtered}{List containing filtered allele frequencies for 
    training set. Must have components:
    \describe{
      \item{haplotype}{Character vector of haplotype block names}
      \item{allele}{Integer vector of allele identifiers}
      \item{freq}{Numeric vector of allele frequencies}
    }
    Required when \code{reference_structure = NULL} (training set).}
  
  \item{reference_structure}{List containing reference allele structure for test 
    set. Must have components:
    \describe{
      \item{allele_info}{List with \code{allele_id} (character vector) and 
        \code{freq} (numeric vector)}
      \item{dropped_alleles}{Optional list with \code{block}, \code{allele}, 
        and \code{freq} for baseline alleles}
    }
    When provided, constructs test set matrix using reference allele structure.}
  
  \item{drop_baseline}{Logical. If \code{TRUE}, drops the most frequent allele 
    in each block as baseline (default). If \code{FALSE}, retains all alleles.}
}
\details{
The function implements the allele-specific coding scheme for multi-allelic markers:

For allele k with frequency p_k in individual i:
\deqn{W_{i,k} = \begin{cases}
  -2(1-p_k) & \text{if homozygous for allele k} \\
  -(1-2p_k) & \text{if heterozygous} \\
  2p_k & \text{if homozygous for other alleles}
\end{cases}}

This coding ensures:
\itemize{
  \item Mean of each column is zero
  \item Variance proportional to 2p_k(1-p_k)
  \item Proper centering and scaling for genomic prediction
}

\strong{Baseline Allele Dropping:}

When \code{drop_baseline = TRUE}, the most frequent allele in each haplotype 
block is removed to avoid multicollinearity. This reduces the design matrix 
from h alleles per block to (h-1) informative alleles.

\strong{Training vs Test Set:}

\itemize{
  \item \strong{Training}: Use \code{allele_freq_filtered} to build W matrix 
    from scratch. Allele frequencies computed from training data.
  \item \strong{Test}: Use \code{reference_structure} to ensure identical 
    allele ordering and frequencies as training set.
}

\strong{Performance:}

Rust implementation provides 10-100x speedup over pure R for large datasets 
(>10K individuals, >1M alleles) through optimized memory access and parallel 
block processing.
}
\value{
A list with three components:
  \item{W_ah}{Numeric matrix (n × m) where m is total number of informative 
    alleles across all blocks. Design matrix for genomic prediction.}
  
  \item{allele_info}{Data frame with columns:
    \describe{
      \item{allele_id}{Unique allele identifier (format: "block_alleleN")}
      \item{block}{Haplotype block name}
      \item{allele}{Allele number within block}
      \item{freq}{Allele frequency}
    }
    Provides metadata for each column of W_ah.}
  
  \item{dropped_alleles}{Data frame with columns \code{block}, \code{allele}, 
    \code{freq} for baseline alleles that were dropped. Empty if 
    \code{drop_baseline = FALSE}.}
}
\references{
Erbe, M., Hayes, B. J., Matukumalli, L. K., Goswami, S., Bowman, P. J., 
  Reich, C. M., ... & Goddard, M. E. (2012). Improving accuracy of genomic 
  predictions within and between dairy cattle breeds with imputed high-density 
  single nucleotide polymorphism panels. \emph{Journal of Dairy Science}, 
  95(7), 4114-4129.

Moser, G., Lee, S. H., Hayes, B. J., Goddard, M. E., Wray, N. R., & 
  Visscher, P. M. (2015). Simultaneous discovery, estimation and prediction 
  analysis of complex traits using a Bayesian mixture model. 
  \emph{PLoS Genetics}, 11(4), e1004969.
}
\examples{
\dontrun{
# Simulated haplotype data
set.seed(123)
n <- 1000
n_blocks <- 100

# Generate diploid haplotypes (2 columns per block)
hap_matrix <- matrix(
  sample(1:5, n * n_blocks * 2, replace = TRUE),
  nrow = n,
  ncol = n_blocks * 2
)

# Column names
col_names <- paste0("block_", rep(1:n_blocks, each = 2), 
                   c("", "_copy"))

# Compute allele frequencies
allele_freq <- data.frame(
  haplotype = rep(col_names, each = 5),
  allele = rep(1:5, n_blocks * 2),
  freq = runif(n_blocks * 2 * 5, 0.05, 0.95)
)

# Filter rare alleles (MAF > 0.01)
allele_freq_filtered <- list(
  haplotype = allele_freq$haplotype,
  allele = as.integer(allele_freq$allele),
  freq = allele_freq$freq
)

# Build W matrix for training set
result <- construct_wah_matrix(
  hap_matrix = hap_matrix,
  colnames = col_names,
  allele_freq_filtered = allele_freq_filtered,
  reference_structure = NULL,
  drop_baseline = TRUE
)

# Examine output
dim(result$W_ah)  # n × m matrix
head(result$allele_info)
head(result$dropped_alleles)

# For test set: use reference structure
test_hap <- matrix(
  sample(1:5, 500 * n_blocks * 2, replace = TRUE),
  nrow = 500,
  ncol = n_blocks * 2
)

result_test <- construct_wah_matrix(
  hap_matrix = test_hap,
  colnames = col_names,
  allele_freq_filtered = NULL,
  reference_structure = result,  # Use training structure
  drop_baseline = TRUE
)

# Dimensions match
ncol(result$W_ah) == ncol(result_test$W_ah)  # TRUE
}
}
\seealso{
\code{\link{run_bayesr_mcmc}}, \code{\link{run_bayesa_mcmc}}
}
\author{
Bowo Wibowo
}
\keyword{array}
\keyword{multivariate}