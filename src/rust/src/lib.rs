use extendr_api::prelude::*;

mod bayesr;
mod bayesa;
mod utils;
mod types;

use bayesr::BayesRRunner;
use bayesa::BayesARunner;

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

// Macro to generate exports
extendr_module! {
    mod genomicbayes_extendr;
    fn run_bayesr_mcmc;
    fn run_bayesa_mcmc;
}