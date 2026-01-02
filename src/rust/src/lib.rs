use extendr_api::prelude::*;

mod matrix;
mod bayesr;
mod bayesa;
mod bayesr_em;
mod bayesa_em;
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

/// Construct W matrix from haplotype genotypes
///
/// @param hap_matrix Matrix of haplotype genotypes (n x 2*blocks)
/// @param colnames Column names for haplotype matrix
/// @param allele_freq_filtered Dataframe with columns: haplotype, allele, freq
/// @param reference_structure Optional reference structure for test set (NULL for training)
/// @param drop_baseline Whether to drop most frequent allele as baseline
/// @return List with W_ah matrix, allele_info dataframe, and dropped_alleles dataframe
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

/// Run BayesR MCMC sampling
///
/// @param W Training genotype matrix (n x p)
/// @param y Phenotype vector (n)
/// @param WtW_diag Diagonal of W'W (p)
/// @param Wty W'y vector (p)
/// @param pi_vec Mixture proportions (4)
/// @param sigma2_vec Variance components (4)
/// @param sigma2_e_init Initial residual variance
/// @param prior_params List of prior hyperparameters
/// @param mcmc_params List of MCMC parameters
/// @return List containing posterior samples and diagnostics
#[extendr]
fn run_bayesr_mcmc(
    w: RMatrix<f64>,
    y: Vec<f64>,
    wtw_diag: Vec<f64>,
    wty: Vec<f64>,
    pi_vec: Vec<f64>,
    sigma2_vec: Vec<f64>,
    sigma2_e_init: f64,
    sigma2_ah: f64,
    prior_params: List,
    mcmc_params: List,
    fold_id: i32,
) -> List {
    
    // Extract MCMC parameters
    let n_iter = mcmc_params.dollar("n_iter").unwrap().as_integer().unwrap() as usize;
    let n_burn = mcmc_params.dollar("n_burn").unwrap().as_integer().unwrap() as usize;
    let n_thin = mcmc_params.dollar("n_thin").unwrap().as_integer().unwrap() as usize;
    let seed = mcmc_params.dollar("seed").unwrap().as_integer().unwrap() as u64;
    
    // Extract prior parameters
    let a0_e = prior_params.dollar("a0_e").unwrap().as_real().unwrap();
    let b0_e = prior_params.dollar("b0_e").unwrap().as_real().unwrap();
    let a0_small = prior_params.dollar("a0_small").unwrap().as_real().unwrap();
    let b0_small = prior_params.dollar("b0_small").unwrap().as_real().unwrap();
    let a0_medium = prior_params.dollar("a0_medium").unwrap().as_real().unwrap();
    let b0_medium = prior_params.dollar("b0_medium").unwrap().as_real().unwrap();
    let a0_large = prior_params.dollar("a0_large").unwrap().as_real().unwrap();
    let b0_large = prior_params.dollar("b0_large").unwrap().as_real().unwrap();
    
    // Convert R matrix to ndarray
    let w_array = utils::rmatrix_to_array2(&w);
    
    // Create runner
    let mut runner = BayesRRunner::new(
        w_array,
        y,
        wtw_diag,
        wty,
        pi_vec,
        sigma2_vec,
        sigma2_e_init,
        sigma2_ah,
        a0_e, b0_e,
        a0_small, b0_small,
        a0_medium, b0_medium,
        a0_large, b0_large,
        n_iter,
        n_burn,
        n_thin,
        seed,
        fold_id,
    );
    
    // Run MCMC
    let results = runner.run();
    
    // Convert ndarray results to R objects
    list!(
        beta_samples = array2_to_rmatrix(&results.beta_samples),
        gamma_samples = array2_to_rmatrix(&results.gamma_samples),
        sigma2_e_samples = array1_to_vec(&results.sigma2_e_samples),
        sigma2_small_samples = array1_to_vec(&results.sigma2_small_samples),
        sigma2_medium_samples = array1_to_vec(&results.sigma2_medium_samples),
        sigma2_large_samples = array1_to_vec(&results.sigma2_large_samples),
        pi_samples = array2_to_rmatrix(&results.pi_samples)
    )
}

/// Run BayesA MCMC sampling
///
/// @param W Training genotype matrix (n x p)
/// @param y Phenotype vector (n)
/// @param WtW_diag Diagonal of W'W (p)
/// @param Wty W'y vector (p)
/// @param nu Degrees of freedom for marker variance prior
/// @param S_squared Prior scale for marker variances
/// @param sigma2_e_init Initial residual variance
/// @param prior_params List of prior hyperparameters
/// @param mcmc_params List of MCMC parameters
/// @return List containing posterior samples and diagnostics
#[extendr]
fn run_bayesa_mcmc(
    w: RMatrix<f64>,
    y: Vec<f64>,
    wtw_diag: Vec<f64>,
    wty: Vec<f64>,
    nu: f64,
    s_squared: f64,
    sigma2_e_init: f64,
    prior_params: List,
    mcmc_params: List,
    fold_id: i32,
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
    
    // Create runner
    let mut runner = BayesARunner::new(
        w_array,
        y,
        wtw_diag,
        wty,
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
    );
    
    // Run MCMC
    let results = runner.run();
    
    // Convert ndarray results to R objects
    list!(
        beta_samples = array2_to_rmatrix(&results.beta_samples),
        sigma2_j_samples = array2_to_rmatrix(&results.sigma2_j_samples),
        sigma2_e_samples = array1_to_vec(&results.sigma2_e_samples)
    )
}

#[extendr]
fn run_bayesr_em(
    w: RMatrix<f64>,
    y: Vec<f64>,
    wtw_diag: Vec<f64>,
    wty: Vec<f64>,
    pi_vec: Vec<f64>,
    sigma2_vec: Vec<f64>,
    sigma2_e_init: f64,
    em_params: List,
    fold_id: i32,
) -> List {
    let max_iter = em_params.dollar("max_iter").unwrap().as_integer().unwrap() as usize;
    let tol = em_params.dollar("tol").unwrap().as_real().unwrap();
    let seed = em_params.dollar("seed")
        .and_then(|s| s.as_integer().ok())
        .unwrap_or(123) as u64;
    
    let w_array = utils::rmatrix_to_array2(&w);
    
    let mut runner = BayesREM::new(
        w_array, y, wtw_diag, wty,
        pi_vec, sigma2_vec, sigma2_e_init,
        max_iter, tol, seed, fold_id,
    );
    
    let results = runner.run();
    
    list!(
        beta_samples = array2_to_rmatrix(&results.beta_samples),
        gamma_samples = array2_to_rmatrix(&results.gamma_samples),
        sigma2_e_samples = array1_to_vec(&results.sigma2_e_samples),
        sigma2_small_samples = array1_to_vec(&results.sigma2_small_samples),
        sigma2_medium_samples = array1_to_vec(&results.sigma2_medium_samples),
        sigma2_large_samples = array1_to_vec(&results.sigma2_large_samples),
        pi_samples = array2_to_rmatrix(&results.pi_samples)
    )
}

#[extendr]
fn run_bayesa_em(
    w: RMatrix<f64>,
    y: Vec<f64>,
    wtw_diag: Vec<f64>,
    wty: Vec<f64>,
    nu: f64,
    s_squared: f64,
    sigma2_e_init: f64,
    em_params: List,
    fold_id: i32,
) -> List {
    let max_iter = em_params.dollar("max_iter").unwrap().as_integer().unwrap() as usize;
    let tol = em_params.dollar("tol").unwrap().as_real().unwrap();
    
    let w_array = utils::rmatrix_to_array2(&w);
    
    let mut runner = BayesAEM::new(
        w_array, y, wtw_diag, wty,
        nu, s_squared, sigma2_e_init,
        max_iter, tol, fold_id,
    );
    
    let results = runner.run();
    
    list!(
        beta_samples = array2_to_rmatrix(&results.beta_samples),
        sigma2_j_samples = array2_to_rmatrix(&results.sigma2_j_samples),
        sigma2_e_samples = array1_to_vec(&results.sigma2_e_samples)
    )
}

// Macro to generate exports
extendr_module! {
    mod masbayes_extendr;
    fn run_bayesr_mcmc;
    fn run_bayesa_mcmc;
    fn run_bayesr_em;
    fn run_bayesa_em; 
    fn construct_wah_matrix;
}