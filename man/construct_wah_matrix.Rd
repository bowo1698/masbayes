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
Da, Y. (2015). Multi-allelic haplotype model based on genetic partition for genomic
  prediction and variance component estimation using SNP markers. \emph{BMC Genetics},
  16(144).
}
\examples{
\dontrun{
# Simulated haplotype data
set.seed(123)
n <- 100  # Number of individuals
B <- 10   # Number of haplotype blocks

# Haplotype matrix: n individuals × 2B columns (diploid)
# Each cell contains allele ID (1, 2, or 3)
hap_matrix <- matrix(
  sample(1:3, n * B * 2, replace = TRUE, prob = c(0.6, 0.3, 0.1)),
  nrow = n, ncol = B * 2
)

# Column names: each block has 2 copies (diploid)
colnames <- paste0("block_", rep(1:B, each = 2))

# Allele frequencies (required for W matrix construction)
# Must be a LIST with: haplotype, allele, freq
allele_freq <- list(
  haplotype = rep(paste0("block_", 1:B), each = 3), # can be blk or anything with "_"
  allele = rep(1:3, B),
  freq = rep(c(0.6, 0.3, 0.1), B)
)

# Build Wah matrix for training set
train_Wah <- construct_wah_matrix(
  hap_matrix = hap_matrix,
  colnames = colnames,
  allele_freq_filtered = allele_freq,
  reference_structure = NULL,
  drop_baseline = TRUE
)

# This results in a list of W_ah matrix, allele_info, and dropped_alleles 
names(train_Wah)
dim(train_Wah$W_ah)  # n × m matrix
head(train_Wah$allele_info)
head(train_Wah$dropped_alleles)

# For test set: use reference structure
n_test <- 200
test_hap <- matrix(
  sample(1:3, n_test * B * 2, replace = TRUE, prob = c(0.6, 0.3, 0.1)),
  nrow = n_test, ncol = B * 2
)

test_Wah <- construct_wah_matrix(
  hap_matrix = test_hap,
  colnames = colnames,
  allele_freq_filtered = NULL,
  reference_structure = train_Wah,  # Use training structure
  drop_baseline = TRUE
)

# Dimensions match
ncol(train_Wah$W_ah) == ncol(test_Wah$W_ah)
}
}
\seealso{
\code{\link{run_bayesr_mcmc}}, \code{\link{run_bayesa_mcmc}}
}
\author{
Agus Wibowo
}
\keyword{array}
\keyword{multivariate}