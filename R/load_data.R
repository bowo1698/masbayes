#' Load the bundled demo dataset
#'
#' Returns a small, deterministic genomic dataset bundled with the package,
#' intended for examples, vignettes, and tests.
#'
#' Two scales are bundled: \code{size = "large"} (n=200, p=500,
#' n_qtl=20) and \code{size = "small"} (default; n=100, p=50, n_qtl=5) that run
#' during \code{R CMD check --examples}.
#' Both have the same family structure (10 full-sib families) and the
#' same 12-field shape, so consumer code does not have to branch on size.
#'
#' The data simulates a small breeding-style population: 10 full-sib
#' families (each from one sire-dam founder pair). The founders are kept in
#' the pedigree only (no genotype, no phenotype). A within-family 80/20
#' train/test split is bundled so every test individual has its full-sibs in
#' training.
#'
#' @section On the microhaplotype representation in \code{d$mh}:
#' In production, microhaplotype genotypes are produced by the
#' \pkg{maspipeline} preprocessing pipeline, where phasing with Beagle
#' or Shapeit, conversion to phased haplotype alleles, LD- or window-based
#' block discovery, then per-strand block encoding. The on-disk output is one
#' tab-separated file per chromosome with header
#' \preformatted{
#' ID         hap_1_1 hap_1_1 hap_1_2 hap_1_2 hap_2_1 hap_2_1 hap_2_2 hap_2_2 ...
#' IND00001        1       1       2       2       1       1       2       2 ...
#' IND00002        1       1       1       1       2       2       1       1 ...
#' }
#' Each block contributes 4 columns: \code{hap_<b>_1 hap_<b>_1 hap_<b>_2 hap_<b>_2}
#' (the per-strand block name repeated twice, with per-SNP allele codes 1 or
#' 2 inside). Block coordinates and allele frequencies live in companion
#' CSVs (e.g.
#' \code{mh_info_ld_haploblock_G0/stats/microhaplotype_coordinates.csv}).
#'
#' The \code{d$mh} stored here is a compact in-memory equivalent for
#' demos: instead of per-SNP allele codes, each strand of each block is
#' collapsed into a single integer haplotype id via a base-3 polynomial
#' encoding of its SNP alleles. Shape becomes \eqn{n \times (2 \cdot n_{blocks})}
#' (two columns per block, not four), and the per-block frequency table is
#' bundled as \code{d$allele_freq}. Both representations encode the same
#' biological information; \code{d$mh} skips the file-layout overhead and
#' the recomputation of allele frequencies.
#'
#' @param size character, either \code{"small"} (default) or \code{"large"}.
#'
#' @return A list with the following elements (dimensions shown for
#'   \code{size = "large"} / \code{size = "small"}):
#' \describe{
#'   \item{\code{snp}}{Integer matrix \eqn{n \times p} of biallelic SNP
#'     dosages (values \code{0/1/2}). Rownames are individual IDs
#'     \code{IND001..INDn}; colnames are \code{SNP001..SNPp}.
#'     Dimensions: 200 x 500 (large) / 100 x 50 (small).}
#'   \item{\code{mh}}{Integer matrix \eqn{n \times (2 \cdot n_{blocks})} of
#'     microhaplotype allele codes. Columns alternate strand 1 / strand 2
#'     per block. The \code{attr(mh, "block_id")} attribute maps each column
#'     to its block. Consumable directly by
#'     \code{\link{construct_wah_matrix}()}. Dimensions: 200 x 500 with 250
#'     blocks (large) / 100 x 50 with 25 blocks (small).}
#'   \item{\code{allele_freq}}{List with parallel vectors
#'     \code{haplotype}, \code{allele}, \code{freq} -- the training-style
#'     allele frequency table required by
#'     \code{\link{construct_wah_matrix}(hap, block_id, allele_freq)}
#'     when no \code{reference_structure} is supplied. Pre-computed from
#'     the full bundled \code{mh} matrix.}
#'   \item{\code{pheno}}{Data frame with \code{n} rows. Columns: \code{id},
#'     \code{sex} (factor F/M, balanced 50/50),
#'     \code{y_cont_qtl_snp}, \code{y_cont_qtl_mh} (continuous traits under
#'     two QTL architectures), \code{y_bin_qtl_snp}, \code{y_bin_qtl_mh}
#'     (binary traits, threshold at median), \code{tbv_qtl_snp},
#'     \code{tbv_qtl_mh} (true breeding values).}
#'   \item{\code{pedigree}}{Data frame with 220 (large) or 120 (small) rows:
#'     10 sire founders + 10 dam founders (all NA parents) plus the
#'     offspring with their sire/dam recorded. Columns: \code{id},
#'     \code{sire}, \code{dam}.}
#'   \item{\code{qtl}}{List with \code{snp_idx}, \code{mh_idx},
#'     \code{effects_snp}, \code{effects_mh} (each length 20 for large, 5
#'     for small; effects drawn from \code{rnorm}, unit-normalised).}
#'   \item{\code{meta}}{List with \code{n}, \code{n_snp}, \code{n_blocks},
#'     \code{n_snp_per_block}, \code{n_qtl}, \code{n_families},
#'     \code{n_per_family}, \code{h2_target}, \code{sex_beta_snp},
#'     \code{sex_beta_mh}, \code{seed}, \code{split_seed}, \code{size}.}
#'   \item{\code{family_id}}{Character vector length \code{n}, values
#'     \code{fam_01..fam_10}.}
#'   \item{\code{train_idx}}{Integer vector -- row indices into
#'     \code{snp}/\code{mh}/\code{pheno} for the training set. Length 160
#'     (large) / 80 (small).}
#'   \item{\code{test_idx}}{Integer vector -- row indices for the test set.
#'     Length 40 (large) / 20 (small). Every test individual has its
#'     full-sibs in \code{train_idx}.}
#'   \item{\code{map_snp}}{Data frame with one row per SNP. Columns
#'     \code{SNP} (character), \code{CHROM} (integer 1..5), \code{POS}
#'     (integer base-pair coordinate). Aligns 1-to-1 with the columns
#'     of \code{snp} and with the design matrix from
#'     \code{\link{construct_snp_matrix}()}. Ready to pass as the
#'     \code{map} argument to \code{\link{run_bayesr}()} for GWAS.
#'     Dimensions: 500 x 3 (large) / 50 x 3 (small).}
#'   \item{\code{map_mh}}{Data frame with one row per MH block. Columns
#'     \code{block_id} (character, matches
#'     \code{unique(attr(mh, "block_id"))}), \code{chr}, \code{start_pos},
#'     \code{end_pos}, \code{n_snps} (integers). Schema matches
#'     \code{microhaplotype_coordinates.csv} produced by the
#'     \code{maspipeline} tool, so production pipelines can pass
#'     \code{map_mh} unchanged into \code{\link{run_bayesr}()}.
#'     Physical positions are synthetic (5 chromosomes, 100 kb intra-chr
#'     SNP spacing, 1 Mb chr base offset); each block spans the two
#'     consecutive SNPs that built it. Dimensions: 250 x 5 (large) /
#'     25 x 5 (small).}
#'   \item{\code{multigen}}{Nested list with the 3-generation
#'     forward-prediction extension. Generated by
#'     \code{tools/make_demo_data_3gen.R} (AlphaSimR + optiSel OCS) and
#'     merged into this RDS. See \emph{Multi-generation extension}
#'     below for the sub-fields.}
#' }
#'
#' @section Multi-generation extension (\code{d$multigen}):
#' For forward-prediction examples
#' (\code{examples/05_forward_prediction.R}), \code{d$multigen} bundles a
#' 3-generation breeding lineage:
#' \itemize{
#'   \item Gen-1 phenotyped (training pool).
#'   \item Gen-2 produced by OCS selection on gen-1, one cohort per QTL
#'     architecture (\code{gen2_snp}, \code{gen2_mh}).
#'   \item Gen-3 produced by OCS on gen-2, also per architecture
#'     (\code{gen3_snp}, \code{gen3_mh}).
#' }
#' Two evaluation scenarios are supported:
#' \describe{
#'   \item{Scenario A}{Train on gen-1, predict gen-2. Use
#'     \code{multigen$reference_structure_gen1} when building
#'     \eqn{W_{\alpha h}} for the test cohort so columns align with
#'     training.}
#'   \item{Scenario B}{Train on gen-1 + gen-2 (per architecture), predict
#'     gen-3. Use
#'     \code{multigen$reference_structure_gen1_gen2_snp} or
#'     \code{multigen$reference_structure_gen1_gen2_mh} accordingly.}
#' }
#' Sub-fields under \code{d$multigen}:
#' \describe{
#'   \item{\code{gen1}}{List(\code{snp}, \code{mh}, \code{allele_freq},
#'     \code{pheno}, \code{sire_of}, \code{dam_of}). \code{pheno} carries
#'     both \code{y_cont_qtl_snp} and \code{y_cont_qtl_mh} plus matching
#'     \code{tbv_qtl_*} columns.}
#'   \item{\code{gen2_snp}, \code{gen2_mh}}{Gen-2 cohort lists; same shape
#'     as \code{gen1} but \code{pheno} only carries the
#'     architecture-matching trait + \code{tbv_qtl_*_true} ground truth.
#'     \code{allele_freq} is the gen-1 frequency table (Scenario A
#'     invariant).}
#'   \item{\code{gen3_snp}, \code{gen3_mh}}{Gen-3 cohort lists, same
#'     pattern as gen-2; \code{allele_freq} is the gen-1+gen-2 combined
#'     frequency table (Scenario B reference).}
#'   \item{\code{reference_structure_gen1},
#'         \code{reference_structure_gen1_gen2_snp},
#'         \code{reference_structure_gen1_gen2_mh}}{Full return values of
#'     \code{\link{construct_wah_matrix}()} on the corresponding training
#'     set. Pass as \code{reference_structure} when building the test
#'     \eqn{W_{\alpha h}}.}
#'   \item{\code{qtl}}{List with \code{mh_idx}, \code{effects_mh},
#'     \code{beta_mh} (effect vector anchored to gen-1
#'     \eqn{W_{\alpha h}} columns), and three SNP centring-mean vectors
#'     for Scenario A / B SNP centring.}
#'   \item{\code{pedigree}}{Combined pedigree across founders + gen-1 +
#'     two gen-2 cohorts + two gen-3 cohorts.}
#'   \item{\code{ocs}}{Diagnostics for all four OCS calls
#'     (current/target/realised average kinship and parent
#'     contributions), nested as
#'     \code{ocs$<arch>$<gen1_to_gen2|gen2_to_gen3>}.}
#'   \item{\code{meta}}{Per-cohort sizes, realised \eqn{h^2} per
#'     generation, target \eqn{\Delta F}, seed, and
#'     \code{AlphaSimR}/\code{optiSel}/\code{masbayes} versions used to
#'     build the dataset.}
#' }
#'
#' @examples
#' d <- load_data()
#' str(d, max.level = 1)
#' dim(d$snp)
#' head(d$pheno)
#'
#' # Smaller dataset for fast examples / unit tests:
#' d_small <- load_data("small")
#' dim(d_small$snp)
#'
#' # Multi-generation extension: forward-prediction layout
#' d3 <- d$multigen
#' str(d3, max.level = 1)
#' dim(d3$gen1$mh); dim(d3$gen2_mh$mh); dim(d3$gen3_mh$mh)
#' # Scenario A: train gen-1, predict gen-2 (W_alpha-h aligned via gen-1 ref):
#' \dontrun{
#' W_tr <- d3$reference_structure_gen1$W_ah
#' W_te <- construct_wah_matrix(
#'   d3$gen2_mh$mh, attr(d3$gen2_mh$mh, "block_id"),
#'   reference_structure = d3$reference_structure_gen1
#' )$W_ah
#' }
#'
#' @export
load_data <- function(size = c("small", "large")) {
  size  <- match.arg(size)
  fname <- if (size == "large") "demo_data.rds" else "demo_data_small.rds"
  path  <- system.file("extdata", fname, package = "masbayes")
  if (!nzchar(path)) {
    stop(
      sprintf("%s not found in masbayes inst/extdata. ", fname),
      "If you are developing masbayes, run `Rscript tools/make_demo_data.R` ",
      "from the genomic_prediction/ project root and reinstall the package."
    )
  }
  readRDS(path)
}
