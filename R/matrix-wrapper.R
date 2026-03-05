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
construct_wah_matrix <- function(hap_matrix, 
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
    wrap__construct_wah_matrix,
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

#' Build W_alpha SNP matrix for training set
#' Allele frequencies are calculated from training individuals.
#'
#' @param snp_files Character vector path file per chr
#' @param ind_ids Character vector ID individual training (for subset and sequence validation)
#' @return List with W_alpha matrix and snp_freq (reference for test set)
#' @export
build_w_snp_train <- function(dosage, ind_ids) {

  missing_ids <- setdiff(ind_ids, rownames(dosage))
  if (length(missing_ids) > 0)
    stop("Individuals not found in SNP dosage: ",
         paste(head(missing_ids, 5), collapse = ", "))

  dosage_train <- dosage[ind_ids, , drop = FALSE]
  p <- colMeans(dosage_train, na.rm = TRUE) / 2

  w <- matrix(0, nrow = nrow(dosage_train), ncol = ncol(dosage_train),
              dimnames = list(ind_ids, colnames(dosage_train)))
  for (j in seq_len(ncol(dosage_train))) {
    pj <- p[j]
    w[, j] <- ifelse(dosage_train[, j] == 0,  2 * pj,
               ifelse(dosage_train[, j] == 1,  2 * pj - 1,
                                               2 * pj - 2))
  }
  list(W_snp = w, snp_freq = p)
}

#' Build W_alpha SNP matrix for test set
#' Using allele frequencies from training
#'
#' @param snp_files Character vector path file per chr
#' @param ind_ids Character vector ID individual test
#' @param snp_freq Named numeric vector of allele frequencies from training
#' @return W_alpha matrix (n_test x n_snp)
#' @export
build_w_snp_test <- function(dosage, ind_ids, snp_freq) {

  missing_ids <- setdiff(ind_ids, rownames(dosage))
  if (length(missing_ids) > 0)
    stop("Individuals not found in SNP dosage: ",
         paste(head(missing_ids, 5), collapse = ", "))

  dosage_test <- dosage[ind_ids, names(snp_freq), drop = FALSE]

  w <- matrix(0, nrow = nrow(dosage_test), ncol = ncol(dosage_test),
              dimnames = list(ind_ids, names(snp_freq)))
  for (j in seq_len(ncol(dosage_test))) {
    pj <- snp_freq[j]
    w[, j] <- ifelse(dosage_test[, j] == 0,  2 * pj,
               ifelse(dosage_test[, j] == 1,  2 * pj - 1,
                                              2 * pj - 2))
  }
  w
}