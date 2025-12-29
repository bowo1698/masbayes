#' Construct W Matrix from Haplotype Genotypes (Rust Implementation)
#'
#' Fast construction of design matrix for multi-allelic markers
#'
#' @param hap_matrix Matrix of haplotype genotypes (n x 2*blocks)
#' @param colnames Column names for haplotype matrix
#' @param allele_freq_filtered Dataframe with haplotype, allele, freq
#' @param reference_structure Reference structure for test set (NULL for training)
#' @param drop_baseline Drop most frequent allele as baseline
#' @return List with W_ah, allele_info, dropped_alleles
#' @export
construct_w_matrix_rust <- function(hap_matrix, 
                                    colnames,
                                    allele_freq_filtered = NULL,
                                    reference_structure = NULL,
                                    drop_baseline = TRUE) {
  
  # Ensure matrix is integer
  if(!is.matrix(hap_matrix)) {
    hap_matrix <- as.matrix(hap_matrix)
  }
  storage.mode(hap_matrix) <- "integer"
  
  # Call Rust function
  result <- .Call(
    wrap__construct_w_matrix_rust,
    hap_matrix,
    as.character(colnames),
    allele_freq_filtered,
    reference_structure,
    as.logical(drop_baseline)
  )
  
  # Convert allele_info to dataframe
  if(length(result$allele_info) > 0) {
    result$allele_info <- data.frame(
      allele_id = result$allele_info$allele_id,
      block = result$allele_info$block,
      allele = result$allele_info$allele,
      freq = result$allele_info$freq,
      stringsAsFactors = FALSE
    )
  }
  
  # Convert dropped_alleles to dataframe
  if(length(result$dropped_alleles) > 0) {
    result$dropped_alleles <- data.frame(
      block = result$dropped_alleles$block,
      allele = result$dropped_alleles$allele,
      freq = result$dropped_alleles$freq,
      stringsAsFactors = FALSE
    )
  } else {
    result$dropped_alleles <- data.frame(
      block = character(0),
      allele = integer(0),
      freq = numeric(0)
    )
  }
  
  result
}