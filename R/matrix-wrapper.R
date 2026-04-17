#' Construct W Matrix from Haplotype Genotypes (Rust Implementation)
#'
#' Fast construction of design matrix for multi-allelic markers using
#' eigen basis decomposition. All h alleles per block are used,
#' projected to h-1 dimensional orthogonal subspace via
#' symmetric eigen decomposition of W'W.
#'
#' @param hap_matrix Matrix of haplotype genotypes (n x 2*blocks), integer
#' @param colnames Column names for haplotype matrix (length = 2*blocks)
#' @param allele_freq_filtered Dataframe with columns: haplotype, allele, freq
#'   (required for training set, NULL for test set)
#' @param reference_structure List from training output containing allele_info
#'   and basis_matrices (required for test set, NULL for training set)
#' @return List with:
#'   \item{W_ah}{Projected W matrix (n x sum(h_b - 1))}
#'   \item{allele_info}{Dataframe with allele_id, block, allele, freq for all h alleles}
#'   \item{basis_matrices}{List of list(block_name, basis) — eigen basis V per block,
#'     needed to construct reference_structure for test set}
#' @export
construct_wah_matrix <- function(hap_matrix,
                                  colnames,
                                  allele_freq_filtered = NULL,
                                  reference_structure  = NULL) {
  if (!is.matrix(hap_matrix)) hap_matrix <- as.matrix(hap_matrix)
  storage.mode(hap_matrix) <- "integer"

  result <- .Call(
    wrap__construct_wah_matrix,
    hap_matrix,
    as.character(colnames),
    allele_freq_filtered,
    reference_structure
  )

  if (length(result$allele_info) > 0) {
    result$allele_info <- data.frame(
      allele_id = result$allele_info$allele_id,
      block     = result$allele_info$block,
      allele    = result$allele_info$allele,
      freq      = result$allele_info$freq,
      stringsAsFactors = FALSE
    )
  }

  result
}