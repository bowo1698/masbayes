use extendr_api::prelude::*;
use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Gamma, Normal, Dirichlet};

/// Convert R matrix to ndarray Array2<i32>
pub fn rmatrix_to_array2_i32(rmat: &RMatrix<i32>) -> Array2<i32> {
    let nrow = rmat.nrows();
    let ncol = rmat.ncols();
    let mut arr = Array2::<i32>::zeros((nrow, ncol));
    
    for i in 0..nrow {
        for j in 0..ncol {
            arr[[i, j]] = rmat[[i, j]];
        }
    }
    
    arr
}

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

/// Digamma function (derivative of log-gamma)
/// 
/// Uses asymptotic expansion for large x and recurrence relation for small x
pub fn digamma(x: f64) -> f64 {
    // For very large x, use asymptotic expansion
    if x > 10.0 {
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        return x.ln() - 0.5 * inv - inv2 / 12.0 + inv2 * inv2 / 120.0 
               - inv2 * inv2 * inv2 / 252.0;
    }
    
    // For x < 0.5, use reflection formula
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return digamma(1.0 - x) - pi / (pi * x).tan();
    }
    
    // For 0.5 <= x < 10, use recurrence: digamma(x+1) = digamma(x) + 1/x
    if x < 10.0 {
        return digamma(x + 1.0) - 1.0 / x;
    }
    
    0.0
}

/// Log-gamma function
/// 
/// Uses Stirling's approximation for large x and recurrence for small x
pub fn lgamma(x: f64) -> f64 {
    const COEFFS: [f64; 8] = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
        0.0,
        0.0,
    ];
    
    if x <= 0.0 {
        return f64::NAN;
    }
    
    // For large x, use Stirling's approximation
    if x > 20.0 {
        let log_sqrt_2pi = 0.91893853320467274178;
        return (x - 0.5) * x.ln() - x + log_sqrt_2pi 
               + ((1.0 / (12.0 * x)) - (1.0 / (360.0 * x * x * x)));
    }
    
    // For small x, use recurrence: lgamma(x+1) = lgamma(x) + ln(x)
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / ((pi * x).sin())).ln() - lgamma(1.0 - x) - x.ln();
    }
    
    // For intermediate x, use Lanczos approximation
    let mut y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    
    let mut ser = 1.000000000190015;
    for (_j, &coeff) in COEFFS.iter().enumerate() {
        y += 1.0;
        ser += coeff / y;
    }
    
    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// Compute autocorrelation at specific lag
/// 
/// Helper function for effective_size calculation
fn autocorr_at_lag(samples: &Array1<f64>, lag: usize, mean: f64, var: f64) -> f64 {
    let n = samples.len();
    if lag >= n || var < 1e-10 {
        return 0.0;
    }
    
    let mut num = 0.0;
    for i in 0..(n - lag) {
        num += (samples[i] - mean) * (samples[i + lag] - mean);
    }
    
    num / ((n - lag) as f64 * var)
}

/// Calculate effective sample size (ESS)
///
/// Uses initial positive sequence estimator with monotone sequence
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
    
    // Calculate initial positive sequence
    let max_lag = (n / 2).min(500);
    let mut rho_sum = 0.0;
    let mut prev_rho = 1.0;
    
    for lag in 1..max_lag {
        let rho = autocorr_at_lag(samples, lag, mean, var);
        
        // Stop if autocorrelation becomes negative or too small
        if rho < 0.0 || rho > prev_rho {
            break;
        }
        
        // Stop if autocorrelation is negligible
        if rho.abs() < 0.05 {
            break;
        }
        
        rho_sum += rho;
        prev_rho = rho;
    }
    
    let ess = n as f64 / (1.0 + 2.0 * rho_sum);
    
    // ESS should be between 1 and n
    ess.max(1.0).min(n as f64)
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
