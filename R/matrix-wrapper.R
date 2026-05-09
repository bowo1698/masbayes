#' Construct the W_ah Design Matrix from Phased Haplotype Genotypes
#'
#' Build the multi-allelic design matrix \eqn{W_{\alpha h}} (Da, 2015) from
#' phased haplotype calls. The matrix is the input expected by
#' \code{\link{run_bayesr}} and \code{\link{run_bayesa}}.
#'
#' @details
#' The haplotype matrix has dimensions \code{n x 2*blocks}: each individual
#' contributes two phased copies per block. Column names encode block
#' identity (e.g. \code{hap_1_1} and \code{hap_1_1_1} are the two copies of
#' block 1).
#'
#' For each block, the most-frequent allele (the baseline, dropped when
#' \code{drop_baseline = TRUE}) is removed for identifiability; the remaining
#' alleles become columns of \eqn{W_{\alpha h}} encoded with the Da (2015)
#' rule. A frequency-weighted projection enforces the sum-to-zero
#' constraint \eqn{E[W_k] = 0}.
#'
#' \strong{Cross-validation / train-test split workflow:}
#' \enumerate{
#'   \item Build the matrix on the \emph{training} set: pass
#'     \code{allele_freq_filtered = <freq_table>} and
#'     \code{reference_structure = NULL}.
#'   \item Build the matrix on the \emph{test} set with the training
#'     structure: pass the entire training-set return value as
#'     \code{reference_structure} and leave \code{allele_freq_filtered = NULL}.
#'     This guarantees the test matrix has identical columns and the same
#'     baseline alleles as the training matrix.
#' }
#'
#' @param hap_matrix Integer matrix of phased haplotype allele codes, with
#'   dimensions \code{n x 2*blocks}. Coerced to integer storage internally.
#' @param colnames Column names of \code{hap_matrix} (length \code{2*blocks}).
#' @param allele_freq_filtered A \code{data.frame} with columns
#'   \code{haplotype}, \code{allele}, and \code{freq}. Required for the
#'   training set; pass \code{NULL} for the test set.
#' @param reference_structure The full return value of a prior training-set
#'   call. Required for the test set; pass \code{NULL} for the training set.
#' @param drop_baseline If \code{TRUE} (default) the most-frequent allele per
#'   block is dropped to ensure full rank.
#'
#' @return A list with three elements:
#' \describe{
#'   \item{\code{W_ah}}{Numeric matrix \code{n x p} with named columns
#'     \code{<block>_allele<k>}.}
#'   \item{\code{allele_info}}{Data frame describing each retained column
#'     (\code{allele_id, block, allele, freq}).}
#'   \item{\code{dropped_alleles}}{Data frame of baseline alleles that were
#'     removed; empty if \code{drop_baseline = FALSE}.}
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n_ind              <- 30
#' n_block            <- 10
#' n_allele_per_block <- 3
#'
#' # Simulate a phased haplotype matrix (n x 2*blocks).
#' # Each pair of columns holds the two phased copies of one block.
#' hap <- matrix(
#'   sample.int(n_allele_per_block, n_ind * n_block * 2, replace = TRUE),
#'   nrow = n_ind
#' )
#' colnames(hap) <- paste0("hap_", rep(seq_len(n_block), each = 2))
#'
#' # Allele frequency table for the training set
#' freq <- data.frame(
#'   haplotype = paste0("hap_",
#'                      rep(seq_len(n_block), each = n_allele_per_block)),
#'   allele    = rep(seq_len(n_allele_per_block), n_block),
#'   freq      = rep(c(0.5, 0.3, 0.2), n_block)
#' )
#'
#' # Training set
#' train   <- construct_wah_matrix(hap, colnames(hap), freq)
#' W_train <- train$W_ah
#'
#' # Test set: reuse the training allele structure
#' hap_te <- matrix(
#'   sample.int(n_allele_per_block, 10 * n_block * 2, replace = TRUE),
#'   nrow = 10
#' )
#' colnames(hap_te) <- colnames(hap)
#' test   <- construct_wah_matrix(hap_te, colnames(hap_te),
#'                                reference_structure = train)
#' W_test <- test$W_ah
#' stopifnot(ncol(W_train) == ncol(W_test))
#' }
#'
#' @seealso \code{\link{run_bayesr}}, \code{\link{run_bayesa}}
#' @references
#' Da, Y. (2015). Multi-allelic haplotype model based on genetic
#' partition for genomic prediction and variance component estimation
#' using SNP markers. \emph{BMC Genetics}, 16(1), 144.
#' \doi{10.1186/s12863-015-0301-1}
#'
#' @export
construct_wah_matrix <- function(hap_matrix,
                                 colnames,
                                 allele_freq_filtered = NULL,
                                 reference_structure  = NULL,
                                 drop_baseline        = TRUE) {

  if (!is.matrix(hap_matrix)) {
    hap_matrix <- as.matrix(hap_matrix)
  }
  storage.mode(hap_matrix) <- "integer"

  result <- .Call(
    wrap__construct_wah_matrix,
    hap_matrix,
    as.character(colnames),
    allele_freq_filtered,
    reference_structure,
    as.logical(drop_baseline)
  )

  if (length(result$allele_info) > 0) {
    result$allele_info <- data.frame(
      allele_id        = result$allele_info$allele_id,
      block            = result$allele_info$block,
      allele           = result$allele_info$allele,
      freq             = result$allele_info$freq,
      stringsAsFactors = FALSE
    )
  }

  if (length(result$dropped_alleles) > 0) {
    result$dropped_alleles <- data.frame(
      block            = result$dropped_alleles$block,
      allele           = result$dropped_alleles$allele,
      freq             = result$dropped_alleles$freq,
      stringsAsFactors = FALSE
    )
  } else {
    result$dropped_alleles <- data.frame(
      block  = character(0),
      allele = integer(0),
      freq   = numeric(0)
    )
  }

  result
}


#' Construct a Centered SNP Design Matrix
#'
#' Build a centered SNP design matrix \eqn{W_{ij} = X_{ij} - 2 p_j} from an
#' allele-dosage matrix \eqn{X} (entries 0, 1, 2). The output is suitable
#' for direct use with \code{\link{run_bayesr}} and
#' \code{\link{run_bayesa}}, which are agnostic to whether the design
#' matrix encodes biallelic SNPs or multi-allelic haplotypes.
#'
#' @details
#' \strong{Training / test workflow.} Always center the test set with
#' \emph{training} allele frequencies, never with frequencies recomputed
#' from the test set itself:
#' \enumerate{
#'   \item Training: call \code{construct_snp_matrix(X_train)} (leave
#'     \code{ref_freq = NULL}). The returned \code{$freq} contains the
#'     per-SNP reference frequencies \eqn{p_j = mean(X_{train,j}) / 2}.
#'   \item Test: call \code{construct_snp_matrix(X_test, ref_freq =
#'     train$freq)} so the columns of \code{W_test} are aligned with
#'     \code{W_train} on the same baseline frequencies.
#' }
#'
#' For phased multi-allelic haplotype data, use
#' \code{\link{construct_wah_matrix}} instead. For other marker
#' representations, build the design matrix manually and pass it to the
#' Bayesian fitters.
#'
#' @param X Numeric or integer matrix of allele dosages
#'   (typically \code{0}, \code{1}, or \code{2}) with dimensions
#'   \code{n x p_snp}. Coerced to \code{double} internally.
#' @param ref_freq Optional numeric vector of length \code{ncol(X)}
#'   giving the reference allele frequencies \eqn{p_j} from the training
#'   set. If \code{NULL} (default) the frequencies are computed from
#'   \code{X} itself (training-set use).
#'
#' @return A list with elements:
#' \describe{
#'   \item{\code{W}}{Numeric matrix \code{n x p_snp} of centered dosages
#'     \eqn{W_{ij} = X_{ij} - 2 p_j}.}
#'   \item{\code{freq}}{Numeric vector of length \code{p_snp} with the
#'     allele frequencies used for centering.}
#'   \item{\code{n}, \code{p}}{Number of individuals and SNPs.}
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n_ind <- 30
#' n_snp <- 50
#' maf   <- runif(n_snp, 0.1, 0.5)
#'
#' # Simulate dosage matrix (0/1/2) for training and test sets
#' X_train <- matrix(rbinom(n_ind * n_snp, 2, prob = maf),
#'                   nrow = n_ind)
#' X_test  <- matrix(rbinom(10    * n_snp, 2, prob = maf),
#'                   nrow = 10)
#'
#' # Training: compute frequencies from X_train
#' train   <- construct_snp_matrix(X_train)
#' W_train <- train$W
#'
#' # Test: reuse training frequencies (do NOT recompute from X_test)
#' test    <- construct_snp_matrix(X_test, ref_freq = train$freq)
#' W_test  <- test$W
#'
#' stopifnot(ncol(W_train) == ncol(W_test))
#'
#' # The returned W can be passed directly to run_bayesr() or run_bayesa()
#' # together with the response y_train.
#' }
#'
#' @seealso \code{\link{construct_wah_matrix}}, \code{\link{run_bayesr}},
#'   \code{\link{run_bayesa}}
#'
#' @export
construct_snp_matrix <- function(X, ref_freq = NULL) {

  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"

  if (is.null(ref_freq)) {
    p_freq <- colMeans(X) / 2
  } else {
    if (length(ref_freq) != ncol(X)) {
      stop(sprintf(
        "length(ref_freq) = %d does not match ncol(X) = %d",
        length(ref_freq), ncol(X)
      ))
    }
    p_freq <- as.numeric(ref_freq)
  }

  W <- sweep(X, 2, 2 * p_freq, "-")

  list(
    W    = W,
    freq = p_freq,
    n    = nrow(X),
    p    = ncol(X)
  )
}
