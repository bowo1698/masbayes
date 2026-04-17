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
use crate::matrix::{WMatrixBuilder, AlleleFreq, ReferenceStructure};

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

    // Parse basis_matrices: list of list(block_name, matrix)
    let mut basis_matrices: Vec<(String, ndarray::Array2<f64>)> = Vec::new();
    if let Ok(basis_list) = ref_list.dollar("basis_matrices") {
        let basis_list: List = basis_list.try_into()
            .unwrap_or_else(|_| List::new(0));
        for item in basis_list.iter() {
            let item_list: List = match item.1.try_into() {
                Ok(l) => l,
                Err(_) => continue,
            };
            let block_name = match item_list.dollar("block_name")
                .ok()
                .and_then(|r| r.as_str().map(|s| s.to_string())) {
                Some(s) => s,
                None => continue,
            };
            let rmat: RMatrix<f64> = match item_list.dollar("basis")
                .ok()
                .and_then(|r| r.try_into().ok()) {
                Some(m) => m,
                None => continue,
            };
            let arr = utils::rmatrix_to_array2(&rmat);
            basis_matrices.push((block_name, arr));
        }
    }

    Ok(ReferenceStructure {
        allele_ids: allele_ids.iter().map(|s| s.to_string()).collect(),
        frequencies,
        basis_matrices,
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
/// @return List with W_ah matrix, allele_info dataframe, and basis_matrices
#[extendr]
fn construct_wah_matrix(
    hap_matrix: RMatrix<i32>,
    colnames: Vec<String>,
    allele_freq_filtered: Nullable<List>,
    reference_structure: Nullable<List>,
) -> List {
    let hap_array = crate::utils::rmatrix_to_array2_i32(&hap_matrix);

    // Test set: pakai reference structure
    if let NotNull(ref_list) = reference_structure {
        let reference = parse_reference_structure(ref_list)
            .expect("Failed to parse reference structure");

        let w_test = WMatrixBuilder::build_with_reference(
            hap_array,
            colnames,
            &reference,
        );

        let mut blocks = Vec::new();
        let mut alleles = Vec::new();
        for allele_id in &reference.allele_ids {
            if let Some(pos) = allele_id.rfind("_allele") {
                blocks.push(allele_id[..pos].to_string());
                alleles.push(allele_id[pos+7..].parse::<i32>().unwrap_or(0));
            } else {
                blocks.push(String::new());
                alleles.push(0);
            }
        }

        let w_test_rmatrix = array2_to_rmatrix(&w_test);

        // Basis matrices tidak perlu dikembalikan untuk test set
        return list!(
            W_ah = w_test_rmatrix,
            allele_info = list!(
                allele_id = reference.allele_ids.clone(),
                block = blocks,
                allele = alleles,
                freq = reference.frequencies.clone()
            ),
            basis_matrices = NULL
        );
    }

    // Training set: build from scratch
    let allele_freq = if let NotNull(freq_list) = allele_freq_filtered {
        parse_allele_freq(freq_list).expect("Failed to parse allele frequencies")
    } else {
        panic!("allele_freq_filtered required for training set");
    };

    let builder = WMatrixBuilder::new(hap_array, colnames, allele_freq);
    let result = builder.build();

    let w_rmatrix = array2_to_rmatrix(&result.w_ah);

    // Serialize basis_matrices sebagai R list of list(block_name, basis)
    let basis_r: Vec<Robj> = result.basis_matrices.iter()
        .map(|(block_name, v)| {
            let v_rmat = array2_to_rmatrix(v);
            list!(
                block_name = block_name.as_str(),
                basis = v_rmat
            ).into_robj()
        })
        .collect();

    list!(
        W_ah = w_rmatrix,
        allele_info = list!(
            allele_id = result.allele_info.iter().map(|a| a.allele_id.clone()).collect::<Vec<_>>(),
            block     = result.allele_info.iter().map(|a| a.block.clone()).collect::<Vec<_>>(),
            allele    = result.allele_info.iter().map(|a| a.allele).collect::<Vec<_>>(),
            freq      = result.allele_info.iter().map(|a| a.freq).collect::<Vec<_>>()
        ),
        basis_matrices = basis_r
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
    sigma2_e_init: f64,
    sigma2_ah: f64,
    prior_params: List,
    mcmc_params: List,
    fold_id: i32,
    is_binary: bool, 
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
        wty,
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
    );
    
    // Run MCMC
    let results = runner.run();

    let z_hat_r = match results.z_hat {
        Some(ref z) => array1_to_vec(z).into_robj(),
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
        beta_hat = array1_to_vec(&results.beta_hat),
        mu_hat = results.mu_hat,
        sigma2_e_hat = results.sigma2_e_hat,
        pred_train = array1_to_vec(&results.pred_train),
        sigma2_g = results.sigma2_g,
        h2 = results.h2,
        z_hat = z_hat_r
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
    is_binary: bool,
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
        is_binary,
    );
    
    // Run MCMC
    let results = runner.run();

    let z_hat_r = match results.z_hat {
        Some(ref z) => array1_to_vec(z).into_robj(),
        None => ().into_robj(),
    };
    
    // Convert ndarray results to R objects
    list!(
        beta_samples = array2_to_rmatrix(&results.beta_samples),
        sigma2_j_samples = array2_to_rmatrix(&results.sigma2_j_samples),
        sigma2_e_samples = array1_to_vec(&results.sigma2_e_samples),
        mu_samples = array1_to_vec(&results.mu_samples),
        beta_hat = array1_to_vec(&results.beta_hat),
        mu_hat = results.mu_hat,
        sigma2_e_hat = results.sigma2_e_hat,
        sigma2_j_hat = array1_to_vec(&results.sigma2_j_hat),
        pred_train = array1_to_vec(&results.pred_train),
        sigma2_g = results.sigma2_g,
        h2 = results.h2,
        z_hat = z_hat_r
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
    //let seed = em_params.dollar("seed")
    //    .ok()                           
    //    .and_then(|s| s.as_integer())   
    //    .unwrap_or(123) as u64;
    
    let w_array = utils::rmatrix_to_array2(&w);
    
    let mut runner = BayesREM::new(
        w_array, y, wtw_diag, wty,
        pi_vec, sigma2_vec, sigma2_e_init,
        max_iter, tol, fold_id, // seed,
    );
    
    let results = runner.run();
    
    list!(
        beta_samples = array2_to_rmatrix(&results.beta_samples),
        gamma_samples = array2_to_rmatrix(&results.gamma_samples),
        sigma2_e_samples = array1_to_vec(&results.sigma2_e_samples),
        sigma2_small_samples = array1_to_vec(&results.sigma2_small_samples),
        sigma2_medium_samples = array1_to_vec(&results.sigma2_medium_samples),
        sigma2_large_samples = array1_to_vec(&results.sigma2_large_samples),
        pi_samples = array2_to_rmatrix(&results.pi_samples),
        mu_samples = array1_to_vec(&results.mu_samples),
        beta_hat = array1_to_vec(&results.beta_hat),
        mu_hat = results.mu_hat,
        sigma2_e_hat = results.sigma2_e_hat,
        pred_train = array1_to_vec(&results.pred_train),
        sigma2_g = results.sigma2_g,
        h2 = results.h2
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
        sigma2_e_samples = array1_to_vec(&results.sigma2_e_samples),
        mu_samples = array1_to_vec(&results.mu_samples),
        beta_hat = array1_to_vec(&results.beta_hat),
        mu_hat = results.mu_hat,
        sigma2_e_hat = results.sigma2_e_hat,
        sigma2_j_hat = array1_to_vec(&results.sigma2_j_hat),
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
    fn construct_wah_matrix;
}