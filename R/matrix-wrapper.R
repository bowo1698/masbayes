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
#' \strong{W_ah encoding (Da, 2015).}
#' For individual \eqn{i} with phased alleles \eqn{(A_i, A_j)} at haplotype
#' block \eqn{h}, the column corresponding to non-baseline allele \eqn{k}
#' (with population frequency \eqn{p_k}) is coded as:
#' \describe{
#'   \item{Homozygous (\eqn{A_i = A_j = k}):}{
#'     \eqn{W_{\alpha h}^{(k)} = -2(1 - p_k)}}
#'   \item{Heterozygous (\eqn{A_i = k} or \eqn{A_j = k}):}{
#'     \eqn{W_{\alpha h}^{(k)} = -(1 - 2p_k)}}
#'   \item{Absent (\eqn{A_i \neq k} and \eqn{A_j \neq k}):}{
#'     \eqn{W_{\alpha h}^{(k)} = 2p_k}}
#' }
#' Rare alleles receive larger absolute deviations; common alleles receive
#' smaller deviations. The population column-mean is zero by construction,
#' ensuring the additive effects sum to zero across individuals.
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
#' d        <- load_data("small")
#' block_id <- attr(d$mh, "block_id")
#'
#' # Training set -- pass d$allele_freq (required when reference_structure is NULL)
#' hap_tr  <- d$mh[d$train_idx, ]
#' train   <- construct_wah_matrix(hap_tr, block_id, d$allele_freq)
#' W_train <- train$W_ah
#'
#' # Test set: reuse the training allele structure to keep columns aligned
#' hap_te <- d$mh[d$test_idx, ]
#' test   <- construct_wah_matrix(hap_te, block_id,
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


#' Construct a SNP Design Matrix
#'
#' Build a SNP design matrix from an allele-dosage matrix \eqn{X} (entries
#' 0, 1, 2). Two encodings are supported: VanRaden centering (default,
#' \eqn{W_{ij} = X_{ij} - 2 p_j}) and per-column z-score standardisation
#' (\eqn{W_{ij} = (X_{ij} - 2 p_j) / s_j}). The output is suitable for
#' direct use with \code{\link{run_bayesr}} and \code{\link{run_bayesa}}.
#'
#' @details
#' \strong{Choosing an encoding.}
#' \itemize{
#'   \item \code{"vanRaden"} (default) — centers each column on the
#'     training-set allele frequency. Column variance is proportional to
#'     \eqn{2 p_j (1 - p_j)}, so rare variants contribute proportionally
#'     less to the implied genomic relationship matrix. This is the
#'     classical genomic-prediction convention and is consistent with
#'     \code{tcrossprod(W) / k_grm} used by GBLUP backends.
#'   \item \code{"zscore"} — additionally divides each column by its
#'     standard deviation, so every marker contributes equal variance
#'     regardless of MAF. Aligned with alternative biallelic-SNP
#'     parameterisations that assume marker-uniform variance contributions.
#' }
#' Both encodings work with either \code{marker_type = "multiallelic"} or
#' \code{"snp"} in the Bayesian fitters, but \code{"zscore"} is intended
#' to pair with \code{marker_type = "snp"} for a fully alternative SNP
#' convention.
#'
#' \strong{Training / test workflow.} Always centre (and scale, for
#' zscore) the test set with \emph{training} statistics, never with
#' statistics recomputed from the test set itself:
#' \enumerate{
#'   \item Training: call \code{construct_snp_matrix(X_train,
#'     encoding = ...)}. The returned \code{$freq} and, for zscore,
#'     \code{$sd} hold the training statistics.
#'   \item Test: call \code{construct_snp_matrix(X_test, encoding = ...,
#'     ref_freq = train$freq, ref_sd = train$sd)} so test columns align
#'     with training columns under the same baseline.
#' }
#'
#' For phased multi-allelic haplotype data, use
#' \code{\link{construct_wah_matrix}} instead.
#'
#' @param X Numeric or integer matrix of allele dosages
#'   (typically \code{0}, \code{1}, or \code{2}) with dimensions
#'   \code{n x p_snp}. Coerced to \code{double} internally.
#' @param encoding Either \code{"vanRaden"} (default) or \code{"zscore"}.
#'   Controls whether columns are additionally divided by their standard
#'   deviation after centering.
#' @param ref_freq Optional numeric vector of length \code{ncol(X)}
#'   giving the reference allele frequencies \eqn{p_j} from the training
#'   set. If \code{NULL} (default) frequencies are computed from
#'   \code{X} itself (training-set use).
#' @param ref_sd Optional numeric vector of length \code{ncol(X)} giving
#'   the training-set column standard deviations. Required only when
#'   \code{encoding = "zscore"} and \code{X} is a test set. If
#'   \code{NULL} for a training call, the sd is computed from \code{X}.
#'   Ignored when \code{encoding = "vanRaden"}.
#'
#' @return A list with elements:
#' \describe{
#'   \item{\code{W}}{Numeric matrix \code{n x p_snp}. For vanRaden:
#'     \eqn{W_{ij} = X_{ij} - 2 p_j}. For zscore:
#'     \eqn{W_{ij} = (X_{ij} - 2 p_j) / s_j}.}
#'   \item{\code{freq}}{Numeric vector of length \code{p_snp} with the
#'     allele frequencies used for centering.}
#'   \item{\code{sd}}{Numeric vector of length \code{p_snp} with the
#'     column standard deviations used for scaling. Present only when
#'     \code{encoding = "zscore"}.}
#'   \item{\code{n}, \code{p}}{Number of individuals and SNPs.}
#' }
#'
#' @examples
#' \dontrun{
#' d <- load_data("small")
#' X_train <- d$snp[d$train_idx, ]
#' X_test  <- d$snp[d$test_idx, ]
#'
#' # vanRaden (default)
#' train_v <- construct_snp_matrix(X_train)
#' test_v  <- construct_snp_matrix(X_test, ref_freq = train_v$freq)
#'
#' # zscore (alternative SNP convention)
#' train_z <- construct_snp_matrix(X_train, encoding = "zscore")
#' test_z  <- construct_snp_matrix(X_test, encoding = "zscore",
#'                                 ref_freq = train_z$freq,
#'                                 ref_sd   = train_z$sd)
#'
#' stopifnot(ncol(train_z$W) == ncol(test_z$W))
#' }
#'
#' @seealso \code{\link{construct_wah_matrix}}, \code{\link{run_bayesr}},
#'   \code{\link{run_bayesa}}
#'
#' @export
construct_snp_matrix <- function(X,
                                 encoding = c("vanRaden", "zscore"),
                                 ref_freq = NULL,
                                 ref_sd   = NULL) {

  encoding <- match.arg(encoding)

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

  if (encoding == "vanRaden") {
    # Vanraden Centering
    W <- sweep(X, 2, 2 * p_freq, "-")
    return(list(
      W    = W,
      freq = p_freq,
      n    = nrow(X),
      p    = ncol(X)
    ))
  }

  # encoding == "zscore"
  if (is.null(ref_sd)) {
    p_sd <- apply(X, 2, sd)
  } else {
    if (length(ref_sd) != ncol(X)) {
      stop(sprintf(
        "length(ref_sd) = %d does not match ncol(X) = %d",
        length(ref_sd), ncol(X)
      ))
    }
    p_sd <- as.numeric(ref_sd)
  }

  bad <- which(!is.finite(p_sd) | p_sd < 1e-5)
  if (length(bad) > 0) {
    stop(sprintf(
      "encoding = 'zscore' requires all columns to have sd >= 1e-5; %d column(s) violate (first index: %d). Filter zero-variance columns upstream.",
      length(bad), bad[1]
    ))
  }

  W <- sweep(sweep(X, 2, 2 * p_freq, "-"), 2, p_sd, "/")
  list(
    W    = W,
    freq = p_freq,
    sd   = p_sd,
    n    = nrow(X),
    p    = ncol(X)
  )
}
