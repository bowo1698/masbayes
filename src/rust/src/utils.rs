use extendr_api::prelude::*;
use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Gamma, Normal, Dirichlet};

/// Convert R matrix to ndarray Array2
pub fn rmatrix_to_array2(rmat: &RMatrix<f64>) -> Array2<f64> {
    let nrow = rmat.nrows();
    let ncol = rmat.ncols();
    let mut arr = Array2::<f64>::zeros((nrow, ncol));
    
    for i in 0..nrow {
        for j in 0..ncol {
            arr[[i, j]] = rmat[[i, j]];
        }
    }
    
    arr
}

/// Sample from inverse-gamma distribution
///
/// InvGamma(a, b) is equivalent to 1/Gamma(a, 1/b)
pub fn rinvgamma<R: Rng>(rng: &mut R, shape: f64, scale: f64) -> f64 {
    let gamma_dist = Gamma::new(shape, 1.0 / scale).unwrap();
    1.0 / gamma_dist.sample(rng)
}

/// Sample from Dirichlet distribution
pub fn rdirichlet<R: Rng>(rng: &mut R, alpha: &Array1<f64>) -> Array1<f64> {
    let alpha_vec: Vec<f64> = alpha.iter().copied().collect();
    let dirichlet = Dirichlet::new(&alpha_vec).unwrap();
    let sample: Vec<f64> = dirichlet.sample(rng);
    Array1::from_vec(sample)
}

/// Sample from normal distribution
pub fn rnorm<R: Rng>(rng: &mut R, mean: f64, sd: f64) -> f64 {
    let normal = Normal::new(mean, sd).unwrap();
    normal.sample(rng)
}

/// Calculate effective sample size (ESS)
///
/// Simple estimator: n / (1 + 2 * sum of autocorrelations)
pub fn effective_size(samples: &Array1<f64>) -> f64 {
    let n = samples.len();
    if n < 10 {
        return n as f64;
    }
    
    let mean = samples.mean().unwrap();
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    
    if var < 1e-10 {
        return n as f64;
    }
    
    // Calculate autocorrelations up to lag n/2
    let max_lag = n / 2;
    let mut rho_sum = 0.0;
    
    for lag in 1..max_lag {
        let mut num = 0.0;
        for i in 0..(n - lag) {
            num += (samples[i] - mean) * (samples[i + lag] - mean);
        }
        let rho = num / ((n - lag) as f64 * var);
        
        // Stop if autocorrelation becomes negligible
        if rho.abs() < 0.05 {
            break;
        }
        
        rho_sum += rho;
    }
    
    n as f64 / (1.0 + 2.0 * rho_sum)
}

/// Geweke convergence diagnostic Z-score
///
/// Compares means of first 10% and last 50% of chain
pub fn geweke_z(samples: &Array1<f64>) -> f64 {
    let n = samples.len();
    
    if n < 100 {
        return 0.0;
    }
    
    let n1 = n / 10;
    let n2_start = n / 2;
    
    // First segment
    let seg1 = samples.slice(ndarray::s![0..n1]);
    let mean1 = seg1.mean().unwrap();
    let var1 = seg1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 as f64 - 1.0);
    
    // Second segment
    let seg2 = samples.slice(ndarray::s![n2_start..]);
    let mean2 = seg2.mean().unwrap();
    let n2 = seg2.len();
    let var2 = seg2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 as f64 - 1.0);
    
    // Z-score
    let se = (var1 / n1 as f64 + var2 / n2 as f64).sqrt();
    
    if se < 1e-10 {
        return 0.0;
    }
    
    (mean1 - mean2) / se
}

/// Tabulate component assignments
pub fn tabulate(gamma: &Array1<usize>, nbins: usize) -> Vec<usize> {
    let mut counts = vec![0; nbins];
    for &g in gamma.iter() {
        if g < nbins {
            counts[g] += 1;
        }
    }
    counts
}
