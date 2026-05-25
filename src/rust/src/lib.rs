//! # masbayes — Rust kernel
//!
//! Computational backend for the `masbayes` R package: Bayesian genomic
//! prediction models for biallelic SNP and multi-allelic microhaplotype
//! markers. All numerically heavy work (MCMC, stochastic EM, design matrix
//! construction) lives here; the R side is a thin extendr wrapper.
//!
//! ## Module map
//!
//! - [`matrix`] — design-matrix construction:
//!   * `WMatrixBuilder` implements the Da (2015) $W_{\alpha h}$ encoding for
//!     phased multi-allelic haplotype data, followed by a per-locus
//!     frequency-weighted row shrinkage.
//!   * Helper routines for biallelic SNP encoding (VanRaden) are exposed via
//!     the FFI wrappers in this file.
//! - [`bayesa`] — BayesA Gibbs sampler. Per-marker variance follows a scaled
//!   inverse chi-squared prior, leading to a t-shrunk effect distribution.
//!   Supports binary traits via Albert–Chib data augmentation.
//! - [`bayesr`] — BayesR Gibbs sampler. Marker effects follow a four-component
//!   normal mixture (one spike at zero plus three slab components scaled
//!   relative to the genetic variance); mixture proportions are updated with
//!   a Dirichlet–Multinomial step.
//! - [`bayesa_em`] / [`bayesr_em`] — Stochastic EM variants of the two
//!   samplers. Faster but discard posterior uncertainty.
//! - [`utils`] — RNG helpers (inverse-gamma, Dirichlet, normal, tabulate)
//!   and R↔`ndarray` conversion shims used across the kernels.
//! - [`types`] — POD result structs returned to R via the extendr wrappers.
//!
//! ## Reproducibility
//!
//! All samplers take a `seed: u64` initialising a PCG64 RNG (`rand_pcg`).
//! Given the same data, hyperparameters, and seed, the kernels reproduce
//! posterior samples bit-for-bit on the same platform.
//!
//! ## Reference
//!
//! Da, Y. (2015). Multi-allelic haplotype model based on genetic partition
//! for genomic prediction and variance component estimation using SNP
//! markers. *BMC Genetics*, 16:144.

use extendr_api::prelude::*;

mod matrix;
mod bayesr;
mod bayesa;
mod bayesr_em;
mod bayesa_em;
mod bayesr_snp;
mod bayesa_snp;
mod utils;
mod types;

use bayesr::BayesRRunner;
use bayesa::BayesARunner;
use bayesr_em::BayesREM;
use bayesa_em::BayesAEM;
use crate::matrix::{WMatrixBuilder, AlleleFreq, ReferenceStructure, DroppedAllele};

/// Convert R list to AlleleFreq vector
fn parse_allele_freq(freq_df: List) -> Result<Vec<AlleleFreq>> {
    let haplotype = freq_df.dollar("haplotype")?
        .as_string_vector()
        .ok_or_else(|| Error::from("'haplotype' must be character vector"))?;
    
    let allele = freq_df.dollar("allele")?
        .as_integer_vector()
        .ok_or_else(|| Error::from("'allele' must be integer vector"))?;
    
    let freq = freq_df.dollar("freq")?
        .as_real_vector()
        .ok_or_else(|| Error::from("'freq' must be numeric vector"))?;
    
    let mut result = Vec::new();
    for i in 0..haplotype.len() {
        result.push(AlleleFreq {
            haplotype: haplotype[i].to_string(),
            allele: allele[i],
            freq: freq[i],
        });
    }
    
    Ok(result)
}

/// Convert R list to ReferenceStructure
fn parse_reference_structure(ref_list: List) -> Result<ReferenceStructure> {
    let allele_info = ref_list.dollar("allele_info")?;
    
    let allele_ids = allele_info.dollar("allele_id")?
        .as_string_vector()
        .ok_or_else(|| Error::from("'allele_id' must be character vector"))?;
    
    let frequencies = allele_info.dollar("freq")?
        .as_real_vector()
        .ok_or_else(|| Error::from("'freq' must be numeric vector"))?;
    
    // Parse dropped alleles if exists
    let mut dropped = Vec::new();
    if let Ok(dropped_df) = ref_list.dollar("dropped_alleles") {
        // Try to parse all three columns, skip if any fails
        let blocks_opt = dropped_df.dollar("block")
            .ok()
            .and_then(|r| r.as_string_vector());
        let alleles_opt = dropped_df.dollar("allele")
            .ok()
            .and_then(|r| r.as_integer_vector());
        let freqs_opt = dropped_df.dollar("freq")
            .ok()
            .and_then(|r| r.as_real_vector());
        
        if let (Some(blocks), Some(alleles), Some(freqs)) = 
            (blocks_opt, alleles_opt, freqs_opt) {
            for i in 0..blocks.len() {
                dropped.push(DroppedAllele {
                    block: blocks[i].to_string(),
                    allele: alleles[i],
                    freq: freqs[i],
                });
            }
        }
    }
    
    Ok(ReferenceStructure {
        allele_ids: allele_ids.iter().map(|s| s.to_string()).collect(),
        frequencies,
        dropped_alleles: dropped,
    })
}

/// Convert Array2 to RMatrix
fn array2_to_rmatrix(arr: &ndarray::Array2<f64>) -> RMatrix<f64> {
    let (nrow, ncol) = arr.dim();
    let mut rmat = RMatrix::new(nrow, ncol);
    
    for i in 0..nrow {
        for j in 0..ncol {
            rmat[[i, j]] = arr[[i, j]];
        }
    }
    
    rmat
}

/// Convert Array1 to Vec
fn array1_to_vec(arr: &ndarray::Array1<f64>) -> Vec<f64> {
    arr.to_vec()
}

/// Construct the multi-allelic design matrix W_αh from phased haplotype
/// data (low-level binding).
///
/// Internal Rust binding called by the R wrapper
/// `construct_wah_matrix()`. End users should always go through the
/// wrapper, which exposes a cleaner argument list and handles
/// train / test alignment automatically.
///
/// # Pipeline
///
/// 1. **Per-block encoding** — Da (2015) three-value rule for each
///    non-baseline microhaplotype `k` with frequency `p_k`:
///
///    ```text
///    W_{i,k} = −2 (1 − p_k)   if individual i is homozygous for k
///    W_{i,k} = −(1 − 2 p_k)   if heterozygous (one copy of k)
///    W_{i,k} =  2 p_k          if k is absent in i's genotype
///    ```
///
/// 2. **Baseline drop** — the most frequent microhaplotype per block is
///    used as the contrast reference and dropped from the columns.
///
/// 3. **Frequency-weighted row shrinkage** — see
///    [`crate::matrix::frequency_weighted_row_shrinkage`]. Applied per
///    block after encoding.
///
/// # Arguments
///
/// - `hap_matrix`: `(n, 2 · n_blocks)` integer matrix of phased
///   microhaplotype codes. Columns alternate paternal/maternal per
///   block.
/// - `colnames`: column names of `hap_matrix`. Used to derive block
///   identity (`block_1`, `block_1_copy` ⇒ both belong to block 1).
/// - `allele_freq_filtered`: required for **training** input — an R
///   data.frame with columns `haplotype` (character), `allele` (integer),
///   `freq` (numeric). Pass `NULL` for test data.
/// - `reference_structure`: required for **test** input — the return
///   value of a prior training call, carrying the column ordering and
///   the dropped-baseline list so the test encoding aligns with training.
///   Pass `NULL` for training data.
/// - `drop_baseline`: whether to drop the most-frequent microhaplotype
///   per block as baseline. Default `TRUE` in the R wrapper; setting
///   `FALSE` keeps all alleles (rare; produces a rank-deficient W).
///
/// # Returns
///
/// R list with three elements:
/// - `W_ah`: the `(n, p)` design matrix with `p = Σ_b (h_b − 1)` columns
///   (after baseline drop and row shrinkage).
/// - `allele_info`: data frame describing each retained column —
///   `allele_id`, `block`, `allele`, `freq`.
/// - `dropped_alleles`: data frame of baseline microhaplotypes that
///   were excluded.
///
/// # Train / test alignment (critical)
///
/// Always derive `allele_freq_filtered` from the **training** data only,
/// then pass the full training return value as `reference_structure`
/// when encoding the test set. Recomputing allele frequencies from the
/// test set itself produces a different centering and biases GEBVs.
#[extendr]
fn construct_wah_matrix(
    hap_matrix: RMatrix<i32>,
    colnames: Vec<String>,
    allele_freq_filtered: Nullable<List>,
    reference_structure: Nullable<List>,
    drop_baseline: bool,
) -> List {
    
    // Convert to ndarray
    let hap_array = crate::utils::rmatrix_to_array2_i32(&hap_matrix);
    
    // Check if using reference structure (test set)
    if let NotNull(ref_list) = reference_structure {
        let reference = parse_reference_structure(ref_list)
            .expect("Failed to parse reference structure");
        
        let w_test = WMatrixBuilder::build_with_reference(
            hap_array,
            colnames,
            &reference,
        );
        
        // Parse block and allele from allele_ids
        let mut blocks = Vec::new();
        let mut alleles = Vec::new();
        
        for allele_id in &reference.allele_ids {
            // Parse "hap_1_1_allele3" -> block="hap_1_1", allele=3
            if let Some(pos) = allele_id.rfind("_allele") {
                let block = allele_id[..pos].to_string();
                let allele_str = &allele_id[pos+7..];  // Skip "_allele"
                let allele: i32 = allele_str.parse().unwrap_or(0);
                
                blocks.push(block);
                alleles.push(allele);
            } else {
                // Fallback if parsing fails
                blocks.push(String::new());
                alleles.push(0);
            }
        }
        
        // Convert to RMatrix and set column names
        let mut w_test_rmatrix = array2_to_rmatrix(&w_test);
        let _ = w_test_rmatrix.set_attrib("dimnames", list!(NULL, reference.allele_ids.clone()));
        
        // Return structure matching test set expectations
        return list!(
            W_ah = w_test_rmatrix,
            allele_info = list!(
                allele_id = reference.allele_ids.clone(),
                block = blocks,
                allele = alleles,
                freq = reference.frequencies.clone()
            ),
            dropped_alleles = if reference.dropped_alleles.is_empty() {
                list!()
            } else {
                list!(
                    block = reference.dropped_alleles.iter().map(|d| d.block.clone()).collect::<Vec<_>>(),
                    allele = reference.dropped_alleles.iter().map(|d| d.allele).collect::<Vec<_>>(),
                    freq = reference.dropped_alleles.iter().map(|d| d.freq).collect::<Vec<_>>()
                )
            }
        );
    }
    
    // Training set: build from scratch
    let allele_freq = if let NotNull(freq_list) = allele_freq_filtered {
        parse_allele_freq(freq_list).expect("Failed to parse allele frequencies")
    } else {
        panic!("allele_freq_filtered required for training set");
    };
    
    let builder = WMatrixBuilder::new(
        hap_array,
        colnames,
        allele_freq,
        drop_baseline,
    );
    
    let result = builder.build();
    
    // Convert W_ah to RMatrix and set column names
    let mut w_rmatrix = array2_to_rmatrix(&result.w_ah);
    let colnames: Vec<String> = result.allele_info.iter()
        .map(|a| a.allele_id.clone())
        .collect();

    // Set column names using R's colnames<- function
    let _ = w_rmatrix.set_attrib("dimnames", list!(NULL, colnames));

    list!(
        W_ah = w_rmatrix,
        allele_info = list!(
            allele_id = result.allele_info.iter().map(|a| a.allele_id.clone()).collect::<Vec<_>>(),
            block = result.allele_info.iter().map(|a| a.block.clone()).collect::<Vec<_>>(),
            allele = result.allele_info.iter().map(|a| a.allele).collect::<Vec<_>>(),
            freq = result.allele_info.iter().map(|a| a.freq).collect::<Vec<_>>()
        ),
        dropped_alleles = if result.dropped_alleles.is_empty() {
            list!()
        } else {
            list!(
                block = result.dropped_alleles.iter().map(|d| d.block.clone()).collect::<Vec<_>>(),
                allele = result.dropped_alleles.iter().map(|d| d.allele).collect::<Vec<_>>(),
                freq = result.dropped_alleles.iter().map(|d| d.freq).collect::<Vec<_>>()
            )
        }
    )
}

/// Run a full BayesR Gibbs MCMC fit (low-level binding).
///
/// Internal Rust binding called by the R wrapper `run_bayesr()`. End users
/// should always go through the R wrapper, which validates inputs, applies
/// the documented defaults, and post-processes the raw posterior samples
/// into the `masbayes` fit object.
///
/// # Model
///
/// ```text
/// y = 1 · μ + X · α + W · β + ε,   ε ~ N(0, σ²_e · I)
/// β_j | γ_j = c ~ N(0, v_c · σ²_g),   v_c ∈ {0, 1e-4, 1e-3, 1e-2}
/// γ_j        ~ Categorical(π),         γ_j ∈ {1, 2, 3, 4}
/// π          ~ Dirichlet(α₀),          α₀ = (1, 1, 1, 1) by default
/// σ²_g, σ²_e ~ InvGamma(·, ·)
/// ```
///
/// Component 1 is the spike (`v_c = 0`, exactly zero effect); components
/// 2–4 are slabs of increasing variance. Sparsity emerges from the
/// Dirichlet weighting on the spike — the data decide what fraction of
/// markers stay zero.
///
/// # Sampling steps per iteration
///
/// See the inline math comments in [`crate::bayesr`] for the full
/// derivation. Briefly:
/// 1. Albert-Chib augmentation (binary trait only).
/// 2. Fixed effects `α` from conjugate normal.
/// 3. Intercept μ from its normal full conditional.
/// 4. For each marker: mixture allocation `γ_j` via log-sum-exp, then
///    `β_j` from the conditional normal given the chosen component.
/// 5. Residual variance `σ²_e` from inverse-gamma (continuous traits).
/// 6. Genetic-variance scale `σ²_g` from inverse-gamma.
/// 7. Mixture proportions `π` from Dirichlet–Multinomial conjugacy.
///
/// # Arguments
///
/// - `w`: `(n, p)` design matrix.
/// - `y`: length-`n` response. For binary, must be coded `0`/`1`.
/// - `wtw_diag`: precomputed diagonal of `W' W` (length `p`). Avoids
///   recomputing it inside the per-marker loop every iteration.
/// - `x`: optional `(n, c)` fixed-effect design.
/// - `pi_vec`: initial mixture proportions (length 4).
/// - `sigma2_e_init`: initial residual variance.
/// - `sigma2_ah`: initial genetic variance.
/// - `prior_params`: list with `a0_e`, `b0_e`, `a0_g`, `b0_g`,
///   `variance_class` (length-4 vector of relative variances).
/// - `mcmc_params`: list with `n_iter`, `n_burn`, `n_thin`, `seed`.
/// - `fold_id`: fold id used in verbose log prefixes.
/// - `is_binary`: enable Albert-Chib path for binary traits.
/// - `verbose`: print per-iteration diagnostics.
///
/// # Returns
///
/// R list with posterior sample matrices (`beta_samples`, `gamma_samples`,
/// `sigma2_*_samples`, `pi_samples`, `mu_samples`, optional
/// `alpha_samples` / `z_samples`), posterior point estimates, derived
/// quantities (`sigma2_g`, `h2`), and convergence diagnostics (ESS,
/// Geweke z).
///
/// # Reproducibility
///
/// Given the same `w`, `y`, hyperparameters, and `seed`, this function
/// produces bit-for-bit identical posterior samples on the same platform.
#[extendr]
fn run_bayesr_mcmc(
    w: RMatrix<f64>,
    y: Vec<f64>,
    wtw_diag: Vec<f64>,
    x: Nullable<RMatrix<f64>>,
    pi_vec: Vec<f64>,
    sigma2_e_init: f64,
    sigma2_ah: f64,
    prior_params: List,
    mcmc_params: List,
    fold_id: i32,
    is_binary: bool,
    verbose: bool,
) -> List {

    // Extract MCMC parameters
    let n_iter = mcmc_params.dollar("n_iter").unwrap().as_integer().unwrap() as usize;
    let n_burn = mcmc_params.dollar("n_burn").unwrap().as_integer().unwrap() as usize;
    let n_thin = mcmc_params.dollar("n_thin").unwrap().as_integer().unwrap() as usize;
    let seed = mcmc_params.dollar("seed").unwrap().as_integer().unwrap() as u64;

    // Extract prior parameters
    let a0_e = prior_params.dollar("a0_e").unwrap().as_real().unwrap();
    let b0_e = prior_params.dollar("b0_e").unwrap().as_real().unwrap();
    
    // Convert R matrix to ndarray
    let w_array = utils::rmatrix_to_array2(&w);
    let x_array = match x {
        NotNull(xm) => Some(utils::rmatrix_to_array2(&xm)),
        Null => None,
    };

    let variance_class = prior_params.dollar("variance_class").unwrap()
        .as_real_vector()
        .expect("'variance_class' must be numeric vector");
    let a0_g = prior_params.dollar("a0_g").unwrap().as_real().unwrap();
    let b0_g = prior_params.dollar("b0_g").unwrap().as_real().unwrap();

    // Create runner
    let mut runner = BayesRRunner::new(
        w_array,
        y,
        wtw_diag,
        x_array,
        pi_vec,
        variance_class,
        sigma2_e_init,
        sigma2_ah,
        a0_e, b0_e,
        a0_g, b0_g,
        n_iter,
        n_burn,
        n_thin,
        seed,
        fold_id,
        is_binary,
        verbose,
    );

    // Run MCMC
    let results = runner.run();

    let z_hat_r = match results.z_hat {
        Some(ref z) => array1_to_vec(z).into_robj(),
        None => ().into_robj(),
    };
    let alpha_samples_r = match results.alpha_samples {
        Some(ref a) => array2_to_rmatrix(a).into_robj(),
        None => ().into_robj(),
    };
    let alpha_hat_r = match results.alpha_hat {
        Some(ref a) => array1_to_vec(a).into_robj(),
        None => ().into_robj(),
    };

    // Convert ndarray results to R objects
    list!(
        beta_samples = array2_to_rmatrix(&results.beta_samples),
        gamma_samples = array2_to_rmatrix(&results.gamma_samples),
        sigma2_e_samples = array1_to_vec(&results.sigma2_e_samples),
        sigma2_small_samples = array1_to_vec(&results.sigma2_small_samples),
        sigma2_medium_samples = array1_to_vec(&results.sigma2_medium_samples),
        sigma2_large_samples = array1_to_vec(&results.sigma2_large_samples),
        pi_samples = array2_to_rmatrix(&results.pi_samples),
        mu_samples = array1_to_vec(&results.mu_samples),
        alpha_samples = alpha_samples_r,
        beta_hat = array1_to_vec(&results.beta_hat),
        mu_hat = results.mu_hat,
        sigma2_e_hat = results.sigma2_e_hat,
        alpha_hat = alpha_hat_r,
        pred_train = array1_to_vec(&results.pred_train),
        sigma2_g = results.sigma2_g,
        h2 = results.h2,
        z_hat = z_hat_r
    )
}

/// Run a full BayesA Gibbs MCMC fit (low-level binding).
///
/// Internal Rust binding called by the R wrapper `run_bayesa()`. End users
/// should always go through the R wrapper, which validates inputs and
/// post-processes the raw posterior samples into the `masbayes` fit
/// object.
///
/// # Model
///
/// ```text
/// y = 1 · μ + X · α + W · β + ε,    ε ~ N(0, σ²_e · I)
/// β_j | σ²_j ~ N(0, σ²_j)
/// σ²_j       ~ InvChi2(ν, S²),       S² = σ²_β / L (scaled)
/// σ²_e       ~ InvGamma(a₀_e, b₀_e)
/// ```
///
/// Each marker carries its own variance `σ²_j` drawn from a scaled
/// inverse-chi-squared prior. Marginalising `σ²_j` yields a `t_ν`-shrunk
/// effect distribution — this is the defining feature of BayesA vs.
/// ridge regression (which assumes a single common variance for all
/// markers).
///
/// # Sampling steps per iteration
///
/// See [`crate::bayesa`] for the full math; briefly:
/// 1. Albert-Chib augmentation (binary trait only).
/// 2. Fixed effects `α` from conjugate normal full conditional.
/// 3. Intercept μ from its normal full conditional.
/// 4. For each marker `j`: sample `σ²_j` from inverse-gamma, then `β_j`
///    from its normal full conditional given the new variance. Maintain
///    the working residual `yadj` incrementally to avoid recomputing
///    `W β` from scratch.
/// 5. Residual variance `σ²_e` from inverse-gamma (continuous traits).
///
/// # Arguments
///
/// - `w`: `(n, p)` design matrix.
/// - `y`: length-`n` response. For binary, must be coded `0`/`1`.
/// - `wtw_diag`: precomputed diagonal of `W' W` (length `p`).
/// - `x`: optional `(n, c)` fixed-effect design.
/// - `nu`: degrees of freedom of the scaled inverse-chi-squared prior
///   on each `σ²_j`. Default 4.5.
/// - `s_squared`: scale of the same distribution.
/// - `sigma2_e_init`: initial residual variance.
/// - `prior_params`: list with `a0_e`, `b0_e` for the residual variance
///   prior.
/// - `mcmc_params`: list with `n_iter`, `n_burn`, `n_thin`, `seed`.
/// - `fold_id`: fold id used in verbose log prefixes.
/// - `is_binary`: enable Albert-Chib path for binary traits.
/// - `verbose`: print per-iteration diagnostics.
///
/// # Returns
///
/// R list with posterior sample arrays (`beta_samples`, `sigma2_j_samples`,
/// `sigma2_e_samples`, `mu_samples`, optional `alpha_samples` /
/// `z_samples`), posterior means, derived quantities (`sigma2_g`, `h2`),
/// and convergence diagnostics.
///
/// # Reproducibility
///
/// Given identical `w`, `y`, hyperparameters, and `seed`, the function
/// produces bit-for-bit identical posterior samples on the same
/// platform.
#[extendr]
fn run_bayesa_mcmc(
    w: RMatrix<f64>,
    y: Vec<f64>,
    wtw_diag: Vec<f64>,
    x: Nullable<RMatrix<f64>>,
    nu: f64,
    s_squared: f64,
    sigma2_e_init: f64,
    prior_params: List,
    mcmc_params: List,
    fold_id: i32,
    is_binary: bool,
    verbose: bool,
) -> List {
    // Extract MCMC parameters
    let n_iter = mcmc_params.dollar("n_iter").unwrap().as_integer().unwrap() as usize;
    let n_burn = mcmc_params.dollar("n_burn").unwrap().as_integer().unwrap() as usize;
    let n_thin = mcmc_params.dollar("n_thin").unwrap().as_integer().unwrap() as usize;
    let seed = mcmc_params.dollar("seed").unwrap().as_integer().unwrap() as u64;
    
    // Extract prior parameters
    let a0_e = prior_params.dollar("a0_e").unwrap().as_real().unwrap();
    let b0_e = prior_params.dollar("b0_e")
        .ok()
        .and_then(|r| r.as_real())
        .unwrap_or(0.0); 
    
    // Convert R matrix to ndarray
    let w_array = utils::rmatrix_to_array2(&w);
    let x_array = match x {
        NotNull(xm) => Some(utils::rmatrix_to_array2(&xm)),
        Null => None,
    };

    // Create runner
    let mut runner = BayesARunner::new(
        w_array,
        y,
        wtw_diag,
        x_array,
        nu,
        s_squared,
        sigma2_e_init,
        a0_e,
        b0_e,
        n_iter,
        n_burn,
        n_thin,
        seed,
        fold_id,
        is_binary,
        verbose,
    );
    
    // Run MCMC
    let results = runner.run();

    let z_hat_r = match results.z_hat {
        Some(ref z) => array1_to_vec(z).into_robj(),
        None => ().into_robj(),
    };
    let alpha_samples_r = match results.alpha_samples {
        Some(ref a) => array2_to_rmatrix(a).into_robj(),
        None => ().into_robj(),
    };
    let alpha_hat_r = match results.alpha_hat {
        Some(ref a) => array1_to_vec(a).into_robj(),
        None => ().into_robj(),
    };

    // Convert ndarray results to R objects
    list!(
        beta_samples = array2_to_rmatrix(&results.beta_samples),
        sigma2_j_samples = array2_to_rmatrix(&results.sigma2_j_samples),
        sigma2_e_samples = array1_to_vec(&results.sigma2_e_samples),
        mu_samples = array1_to_vec(&results.mu_samples),
        alpha_samples = alpha_samples_r,
        beta_hat = array1_to_vec(&results.beta_hat),
        mu_hat = results.mu_hat,
        sigma2_e_hat = results.sigma2_e_hat,
        sigma2_j_hat = array1_to_vec(&results.sigma2_j_hat),
        alpha_hat = alpha_hat_r,
        pred_train = array1_to_vec(&results.pred_train),
        sigma2_g = results.sigma2_g,
        h2 = results.h2,
        z_hat = z_hat_r
    )
}

/// Run the SNP-mode BayesA Gibbs MCMC fit (low-level binding).
/// Called by `run_bayesa()` when `marker_type = "snp"`. Aggregates
/// `sigma2_g` per iter as `var(W·β_iter)` via a running u accumulator;
/// supports binary traits via Albert-Chib with `sigma2_e` fixed at 1.
/// @export
#[extendr]
fn run_bayesa_snp_mcmc(
    w: RMatrix<f64>,
    y: Vec<f64>,
    wtw_diag: Vec<f64>,
    x: Nullable<RMatrix<f64>>,
    nu: f64,
    s_squared: f64,
    sigma2_e_init: f64,
    prior_params: List,
    mcmc_params: List,
    fold_id: i32,
    is_binary: bool,
    verbose: bool,
) -> List {
    let n_iter = mcmc_params.dollar("n_iter").unwrap().as_integer().unwrap() as usize;
    let n_burn = mcmc_params.dollar("n_burn").unwrap().as_integer().unwrap() as usize;
    let n_thin = mcmc_params.dollar("n_thin").unwrap().as_integer().unwrap() as usize;
    let seed = mcmc_params.dollar("seed").unwrap().as_integer().unwrap() as u64;

    let a0_e = prior_params.dollar("a0_e").unwrap().as_real().unwrap();
    let b0_e = prior_params.dollar("b0_e").unwrap().as_real().unwrap();

    let w_array = utils::rmatrix_to_array2(&w);
    let x_array = match x {
        NotNull(xm) => Some(utils::rmatrix_to_array2(&xm)),
        Null => None,
    };

    let mut runner = bayesa_snp::BayesASNPRunner::new(
        w_array,
        y,
        wtw_diag,
        x_array,
        nu,
        s_squared,
        sigma2_e_init,
        a0_e, b0_e,
        n_iter,
        n_burn,
        n_thin,
        seed,
        fold_id,
        is_binary,
        verbose,
    );

    let results = runner.run();

    let alpha_samples_r = match results.alpha_samples {
        Some(ref a) => array2_to_rmatrix(a).into_robj(),
        None => ().into_robj(),
    };
    let alpha_hat_r = match results.alpha_hat {
        Some(ref a) => array1_to_vec(a).into_robj(),
        None => ().into_robj(),
    };

    let z_hat_r = match results.z_hat {
        Some(ref z) => array1_to_vec(z).into_robj(),
        None => ().into_robj(),
    };

    list!(
        beta_samples = array2_to_rmatrix(&results.beta_samples),
        sigma2_j_samples = array2_to_rmatrix(&results.sigma2_j_samples),
        sigma2_e_samples = array1_to_vec(&results.sigma2_e_samples),
        mu_samples = array1_to_vec(&results.mu_samples),
        alpha_samples = alpha_samples_r,
        beta_hat = array1_to_vec(&results.beta_hat),
        mu_hat = results.mu_hat,
        sigma2_e_hat = results.sigma2_e_hat,
        sigma2_j_hat = array1_to_vec(&results.sigma2_j_hat),
        alpha_hat = alpha_hat_r,
        pred_train = array1_to_vec(&results.pred_train),
        sigma2_g = results.sigma2_g,
        h2 = results.h2,
        z_hat = z_hat_r
    )
}

/// Run a BayesR stochastic-EM fit (low-level binding).
///
/// Internal Rust binding called by the R wrapper `run_bayesr(method = "em")`.
/// Run the SNP-mode BayesR Gibbs MCMC fit (low-level binding).
/// Called by `run_bayesr()` when `marker_type = "snp"`. Softmax mixture
/// indicator over 4 effect classes with a single base variance scaled by
/// `fold[k]`; supports binary traits via Albert-Chib with `sigma2_e` fixed.
/// @export
#[extendr]
fn run_bayesr_snp_mcmc(
    w: RMatrix<f64>,
    y: Vec<f64>,
    wtw_diag: Vec<f64>,
    x: Nullable<RMatrix<f64>>,
    pi_vec: Vec<f64>,
    sigma2_e_init: f64,
    sigma2_ah: f64,
    prior_params: List,
    mcmc_params: List,
    fold_id: i32,
    is_binary: bool,
    verbose: bool,
) -> List {
    // Extract MCMC parameters
    let n_iter = mcmc_params.dollar("n_iter").unwrap().as_integer().unwrap() as usize;
    let n_burn = mcmc_params.dollar("n_burn").unwrap().as_integer().unwrap() as usize;
    let n_thin = mcmc_params.dollar("n_thin").unwrap().as_integer().unwrap() as usize;
    let seed = mcmc_params.dollar("seed").unwrap().as_integer().unwrap() as u64;

    // Extract prior parameters (same shape as MH BayesR kernel)
    let a0_e = prior_params.dollar("a0_e").unwrap().as_real().unwrap();
    let b0_e = prior_params.dollar("b0_e").unwrap().as_real().unwrap();
    let a0_g = prior_params.dollar("a0_g").unwrap().as_real().unwrap();
    let b0_g = prior_params.dollar("b0_g").unwrap().as_real().unwrap();
    let variance_class = prior_params.dollar("variance_class").unwrap()
        .as_real_vector()
        .expect("'variance_class' must be numeric vector");

    // Convert R matrix to ndarray
    let w_array = utils::rmatrix_to_array2(&w);
    let x_array = match x {
        NotNull(xm) => Some(utils::rmatrix_to_array2(&xm)),
        Null => None,
    };

    let mut runner = bayesr_snp::BayesRSNPRunner::new(
        w_array,
        y,
        wtw_diag,
        x_array,
        pi_vec,
        variance_class,
        sigma2_e_init,
        sigma2_ah,
        a0_e, b0_e,
        a0_g, b0_g,
        n_iter,
        n_burn,
        n_thin,
        seed,
        fold_id,
        is_binary,
        verbose,
    );

    let results = runner.run();

    let alpha_samples_r = match results.alpha_samples {
        Some(ref a) => array2_to_rmatrix(a).into_robj(),
        None => ().into_robj(),
    };
    let alpha_hat_r = match results.alpha_hat {
        Some(ref a) => array1_to_vec(a).into_robj(),
        None => ().into_robj(),
    };

    let z_hat_r = match results.z_hat {
        Some(ref z) => array1_to_vec(z).into_robj(),
        None => ().into_robj(),
    };

    list!(
        beta_samples = array2_to_rmatrix(&results.beta_samples),
        gamma_samples = array2_to_rmatrix(&results.gamma_samples),
        sigma2_e_samples = array1_to_vec(&results.sigma2_e_samples),
        sigma2_small_samples = array1_to_vec(&results.sigma2_small_samples),
        sigma2_medium_samples = array1_to_vec(&results.sigma2_medium_samples),
        sigma2_large_samples = array1_to_vec(&results.sigma2_large_samples),
        pi_samples = array2_to_rmatrix(&results.pi_samples),
        mu_samples = array1_to_vec(&results.mu_samples),
        alpha_samples = alpha_samples_r,
        beta_hat = array1_to_vec(&results.beta_hat),
        mu_hat = results.mu_hat,
        sigma2_e_hat = results.sigma2_e_hat,
        alpha_hat = alpha_hat_r,
        pred_train = array1_to_vec(&results.pred_train),
        sigma2_g = results.sigma2_g,
        h2 = results.h2,
        z_hat = z_hat_r
    )
}

/// Replaces the full Gibbs sampling of [`run_bayesr_mcmc`] with an EM-style
/// coordinate-ascent fit that returns point estimates of marker effects
/// and variance components rather than posterior samples.
///
/// # Algorithm overview
///
/// At each iteration, EM performs:
/// 1. **E-step**: compute the soft mixture membership probabilities for
///    every marker (which mixture component each `β_j` likely belongs to)
///    given the current parameters.
/// 2. **M-step**: update marker effects, mixture proportions `π`, the
///    component-specific variances `σ²_c`, the residual variance `σ²_e`,
///    and the intercept μ to maximise the expected complete-data
///    log-likelihood under the soft memberships.
///
/// Compared with the full Gibbs sampler:
///
/// - **No RNG draws**, hence no need for a seed. Two runs on identical
///   inputs produce bit-for-bit identical output.
/// - **Faster per iteration**: no posterior sample storage, no log-sum-exp
///   for stochastic mixture allocation.
/// - **No uncertainty**: only point estimates. Posterior credible
///   intervals, ESS, and Geweke diagnostics are not available.
///
/// # Arguments
///
/// - `w`: `(n, p)` design matrix (W_αh from Da encoding or VanRaden SNP).
/// - `y`: length-`n` response vector. Continuous trait only here; binary
///   support is currently only wired through the Gibbs path.
/// - `wtw_diag`: length-`p` precomputed diagonal of `W' W`, supplied by
///   the R wrapper to avoid recomputing inside the kernel.
/// - `x`: optional `(n, c)` fixed-effects design (Nullable).
/// - `pi_vec`: length-4 initial mixture proportions.
/// - `sigma2_vec`: length-4 initial variance for each mixture component.
/// - `sigma2_e_init`: initial residual variance.
/// - `em_params`: list with `max_iter` and `tol` controlling convergence.
/// - `fold_id`: cross-validation fold id (used for verbose log prefixes).
/// - `verbose`: print per-iteration progress.
///
/// # Returns
///
/// R list with the same shape as `run_bayesr_mcmc` (so the R wrapper can
/// use identical post-processing), but the "sample" arrays now contain the
/// EM trajectory rather than posterior draws.
#[extendr]
fn run_bayesr_em(
    w: RMatrix<f64>,
    y: Vec<f64>,
    wtw_diag: Vec<f64>,
    x: Nullable<RMatrix<f64>>,
    pi_vec: Vec<f64>,
    sigma2_vec: Vec<f64>,
    sigma2_e_init: f64,
    em_params: List,
    fold_id: i32,
    verbose: bool,
) -> List {
    let max_iter = em_params.dollar("max_iter").unwrap().as_integer().unwrap() as usize;
    let tol = em_params.dollar("tol").unwrap().as_real().unwrap();

    let w_array = utils::rmatrix_to_array2(&w);
    let x_array = match x {
        NotNull(xm) => Some(utils::rmatrix_to_array2(&xm)),
        Null => None,
    };

    let mut runner = BayesREM::new(
        w_array, y, wtw_diag, x_array,
        pi_vec, sigma2_vec, sigma2_e_init,
        max_iter, tol, fold_id, verbose,
    );

    let results = runner.run();

    let alpha_samples_r = match results.alpha_samples {
        Some(ref a) => array2_to_rmatrix(a).into_robj(),
        None => ().into_robj(),
    };
    let alpha_hat_r = match results.alpha_hat {
        Some(ref a) => array1_to_vec(a).into_robj(),
        None => ().into_robj(),
    };

    list!(
        beta_samples = array2_to_rmatrix(&results.beta_samples),
        gamma_samples = array2_to_rmatrix(&results.gamma_samples),
        sigma2_e_samples = array1_to_vec(&results.sigma2_e_samples),
        sigma2_small_samples = array1_to_vec(&results.sigma2_small_samples),
        sigma2_medium_samples = array1_to_vec(&results.sigma2_medium_samples),
        sigma2_large_samples = array1_to_vec(&results.sigma2_large_samples),
        pi_samples = array2_to_rmatrix(&results.pi_samples),
        mu_samples = array1_to_vec(&results.mu_samples),
        alpha_samples = alpha_samples_r,
        beta_hat = array1_to_vec(&results.beta_hat),
        mu_hat = results.mu_hat,
        sigma2_e_hat = results.sigma2_e_hat,
        alpha_hat = alpha_hat_r,
        pred_train = array1_to_vec(&results.pred_train),
        sigma2_g = results.sigma2_g,
        h2 = results.h2
    )
}

/// Run a BayesA stochastic-EM fit (low-level binding).
///
/// Internal Rust binding called by the R wrapper `run_bayesa(method = "em")`.
/// The EM variant of BayesA replaces full Gibbs sampling of the per-marker
/// variances `σ²_j` with their conditional expectation (the E-step), then
/// updates marker effects, residual variance, and intercept in closed
/// form (the M-step).
///
/// # Why EM for BayesA
///
/// The per-marker variances `σ²_j` are the defining feature of BayesA —
/// they induce Student-t shrinkage on marker effects. Gibbs samples
/// `σ²_j` from its scaled-inverse-chi-squared full conditional every
/// iteration; EM plugs in the conditional mean instead. The result is:
///
/// - A point estimate of `β` and `σ²_j` rather than posterior samples.
/// - Deterministic output (no RNG).
/// - Typical run time on the order of 1/10 the equivalent Gibbs run.
///
/// Use the EM variant when only point estimates are needed (e.g. GEBV
/// computation inside a CV loop). Use [`run_bayesa_mcmc`] when you need
/// credible intervals, heritability uncertainty, or convergence diagnostics.
///
/// # Arguments
///
/// - `w`, `y`, `wtw_diag`, `x`: same as the Gibbs runner.
/// - `nu`: degrees of freedom of the scaled inverse-chi-squared prior on
///   each `σ²_j`. Default 4.5 (used by the R wrapper).
/// - `s_squared`: prior scale of the same distribution.
/// - `sigma2_e_init`: initial residual variance.
/// - Other arguments: standard EM controls (`em_params`, `fold_id`,
///   `verbose`).
///
/// # Returns
///
/// R list shaped like `run_bayesa_mcmc` for shape compatibility, but the
/// "sample" arrays carry EM trajectory rather than posterior draws.
#[extendr]
fn run_bayesa_em(
    w: RMatrix<f64>,
    y: Vec<f64>,
    wtw_diag: Vec<f64>,
    x: Nullable<RMatrix<f64>>,
    nu: f64,
    s_squared: f64,
    sigma2_e_init: f64,
    em_params: List,
    fold_id: i32,
    verbose: bool,
) -> List {
    let max_iter = em_params.dollar("max_iter").unwrap().as_integer().unwrap() as usize;
    let tol = em_params.dollar("tol").unwrap().as_real().unwrap();

    let w_array = utils::rmatrix_to_array2(&w);
    let x_array = match x {
        NotNull(xm) => Some(utils::rmatrix_to_array2(&xm)),
        Null => None,
    };

    let mut runner = BayesAEM::new(
        w_array, y, wtw_diag, x_array,
        nu, s_squared, sigma2_e_init,
        max_iter, tol, fold_id, verbose,
    );
    
    let results = runner.run();

    let alpha_samples_r = match results.alpha_samples {
        Some(ref a) => array2_to_rmatrix(a).into_robj(),
        None => ().into_robj(),
    };
    let alpha_hat_r = match results.alpha_hat {
        Some(ref a) => array1_to_vec(a).into_robj(),
        None => ().into_robj(),
    };

    list!(
        beta_samples = array2_to_rmatrix(&results.beta_samples),
        sigma2_j_samples = array2_to_rmatrix(&results.sigma2_j_samples),
        sigma2_e_samples = array1_to_vec(&results.sigma2_e_samples),
        mu_samples = array1_to_vec(&results.mu_samples),
        alpha_samples = alpha_samples_r,
        beta_hat = array1_to_vec(&results.beta_hat),
        mu_hat = results.mu_hat,
        sigma2_e_hat = results.sigma2_e_hat,
        sigma2_j_hat = array1_to_vec(&results.sigma2_j_hat),
        alpha_hat = alpha_hat_r,
        pred_train = array1_to_vec(&results.pred_train),
        sigma2_g = results.sigma2_g,
        h2 = results.h2
    )
}

// Macro to generate exports
extendr_module! {
    mod masbayes_extendr;
    fn run_bayesr_mcmc;
    fn run_bayesa_mcmc;
    fn run_bayesr_em;
    fn run_bayesa_em;
    fn run_bayesr_snp_mcmc;
    fn run_bayesa_snp_mcmc;
    fn construct_wah_matrix;
}