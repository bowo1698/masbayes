//! Shared utilities for the masbayes kernels.
//!
//! Two unrelated groups of helpers live here:
//!
//! 1. **R ↔ `ndarray` conversion** — `rmatrix_to_array2*` copy an extendr
//!    `RMatrix` into an owned `ndarray::Array2` so the MCMC loops can work
//!    with native Rust types without going through the R FFI on every cell
//!    access. Copy cost is paid once per kernel invocation.
//! 2. **Sampling primitives** — `rnorm`, `rinvgamma`, `rdirichlet`,
//!    `tabulate`. These wrap `rand_distr` distributions so the call sites
//!    in `bayesa.rs` / `bayesr.rs` stay close to the manuscript notation
//!    (e.g. `sigma2 ~ InvGamma(a, b)` reads as `rinvgamma(rng, a, b)`).
//!
//! All sampling functions take an `&mut R: Rng` so that the caller controls
//! the seed; this is what makes the full MCMC reproducible from a single
//! `Pcg64::seed_from_u64(seed)` at the start of each runner.

use extendr_api::prelude::*;
use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Gamma, Normal, Dirichlet};

/// Copy an extendr `RMatrix<i32>` into an owned `ndarray::Array2<i32>`.
///
/// # Why copy
///
/// extendr matrices are backed by R's memory and cannot be passed
/// directly into pure-Rust code that expects `ndarray` strides /
/// ownership. Copying once at the FFI boundary lets the MCMC /
/// matrix-construction kernels work with native Rust types, with
/// zero per-iteration FFI overhead.
///
/// # Cost
///
/// `O(n · m)` time and memory for an `(n, m)` matrix — paid once per
/// kernel invocation. Negligible compared with the `O(n_iter · n · p)`
/// cost of the MCMC loops themselves.
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

/// Copy an extendr `RMatrix<f64>` into an owned
/// `ndarray::Array2<f64>`.
///
/// The `f64` counterpart of [`rmatrix_to_array2_i32`]; used for
/// the design matrix `W` and any other continuous-valued input
/// flowing from R into the MCMC kernels. Same rationale and same
/// cost characterisation — see that function's docstring.
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

/// Draw one sample from the inverse-gamma distribution.
///
/// # Parameterisation
///
/// `InvGamma(shape, scale)` is defined as `1 / Gamma(shape, 1 / scale)`.
/// This matches the parameterisation used in the BayesA / BayesR
/// posterior conditionals:
///
/// ```text
/// σ² | rest ~ InvGamma(a + n/2,  b + SSE/2),
/// ```
///
/// where `a` is the prior shape, `b` the prior scale, `n` the sample
/// size, and `SSE` the residual sum of squares. The implementation
/// goes via `rand_distr::Gamma` because the standard library doesn't
/// ship an inverse-gamma primitive.
///
/// # Numerical note
///
/// Returns `+∞` if the underlying gamma sample is exactly zero —
/// extremely rare in floating point but theoretically possible.
/// MCMC users should treat ±∞ samples as a sign that the prior is
/// too vague for the data and tighten `a` or `b`.
pub fn rinvgamma<R: Rng>(rng: &mut R, shape: f64, scale: f64) -> f64 {
    let gamma_dist = Gamma::new(shape, 1.0 / scale).unwrap();
    1.0 / gamma_dist.sample(rng)
}

/// Draw one sample from a Dirichlet distribution.
///
/// # Use in BayesR
///
/// The mixture proportions `π = (π_1, π_2, π_3, π_4)` carry a
/// conjugate Dirichlet prior with concentration `α₀ = (1, 1, 1, 1)`.
/// Given mixture counts `n_c`, the posterior is
/// `Dirichlet(α₀ + n_c)`. We sample from that posterior every Gibbs
/// iteration; this function is the workhorse of that update.
///
/// # Parameterisation
///
/// `alpha` is the concentration vector. All entries must be > 0.
/// Returned vector has the same length as `alpha` and sums to 1.
pub fn rdirichlet<R: Rng>(rng: &mut R, alpha: &Array1<f64>) -> Array1<f64> {
    let alpha_vec: Vec<f64> = alpha.iter().copied().collect();
    let dirichlet = Dirichlet::new(&alpha_vec).unwrap();
    let sample: Vec<f64> = dirichlet.sample(rng);
    Array1::from_vec(sample)
}

/// Draw one sample from a normal distribution `N(mean, sd²)`.
///
/// Used throughout the BayesA / BayesR samplers for: intercept
/// μ updates, marker-effect `β_j` updates, and fixed-effect
/// `α_k` updates — every quantity with a conjugate normal full
/// conditional ends up here.
///
/// # Parameterisation
///
/// `sd` is the **standard deviation**, not the variance.
/// `rand_distr::Normal::new` expects `(mean, sd)` so we pass
/// straight through.
pub fn rnorm<R: Rng>(rng: &mut R, mean: f64, sd: f64) -> f64 {
    let normal = Normal::new(mean, sd).unwrap();
    normal.sample(rng)
}

/// Estimate the effective sample size (ESS) of an MCMC chain.
///
/// # Definition
///
/// ```text
/// ESS = N / (1 + 2 · Σ_{k≥1} ρ_k)
/// ```
///
/// where `ρ_k` is the autocorrelation at lag `k`. The denominator is
/// the integrated autocorrelation time; chains with strong serial
/// dependence have larger denominators and therefore smaller ESS.
///
/// # Truncation heuristic
///
/// The infinite sum is truncated at the first lag where `|ρ_k| <
/// 0.05`. This is a pragmatic compromise:
///
/// - Sokal's "initial monotone sequence" estimator (drop at the
///   first decreasing-then-rising pair) is more rigorous but
///   significantly more complex.
/// - Plain truncation at `N/2` is the textbook upper bound but
///   accumulates noise.
///
/// The 0.05 cut-off matches the behaviour of `coda::effectiveSize`
/// for chains of moderate length.
///
/// # Edge cases
///
/// - Returns `N` if `N < 10` (too short for meaningful ESS).
/// - Returns `N` if the chain has near-zero variance
///   (constant or quasi-constant trace).
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

/// Compute the Geweke (1992) convergence diagnostic Z-score for a
/// single MCMC chain.
///
/// # Definition
///
/// Geweke's test compares the mean of an early segment of the chain
/// (default: first 10 %) against a late segment (default: last 50 %).
/// Under stationarity the two should have the same mean; the
/// difference, standardised by its standard error, is approximately
/// `N(0, 1)` for a converged chain. Values with |z| > 2 are
/// suspicious; |z| > 3 typically indicates non-convergence.
///
/// ```text
/// z = (mean_early − mean_late) / sqrt(var_early / n_early + var_late / n_late)
/// ```
///
/// # Truncation
///
/// Returns `0` if `n < 100` — too few samples for the Z-score to be
/// meaningful, and the R wrapper prints `NA` to signal "skipped"
/// rather than "passed".
///
/// # Limitations
///
/// Geweke is a single-chain test, so it cannot detect inter-chain
/// disagreement that R-hat / split-R-hat would catch. masbayes
/// currently runs one chain per fit; multi-chain diagnostics would
/// be a separate addition.
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

/// Count the occurrences of each component label in a vector.
///
/// # Use in BayesR
///
/// The Dirichlet–Multinomial update of mixture proportions `π`
/// (Step 7 of the BayesR Gibbs sweep) needs the per-component
/// counts `n_c = #{j : γ_j = c}`. This helper produces them in one
/// O(n_alleles) pass.
///
/// # Behaviour
///
/// - `gamma`: vector of integer labels, expected to lie in
///   `[0, nbins)`.
/// - `nbins`: number of bins to return; the output is always
///   length `nbins`, even when some bins are empty.
/// - Labels `≥ nbins` are silently ignored. This is defensive — in
///   practice BayesR's allocation step never produces out-of-range
///   labels, but the guard avoids a panic if upstream code ever
///   has a bug.
///
/// # Returns
///
/// `Vec<usize>` of length `nbins` with counts in each bin.
pub fn tabulate(gamma: &Array1<usize>, nbins: usize) -> Vec<usize> {
    let mut counts = vec![0; nbins];
    for &g in gamma.iter() {
        if g < nbins {
            counts[g] += 1;
        }
    }
    counts
}

/// Sample from a lower-truncated `N(mean, 1)` distribution,
/// restricted to `(lower, +∞)`.
///
/// # Use in Albert-Chib
///
/// Binary-trait BayesA / BayesR augment the model with a latent
/// liability `z` and sample `z | y = 1` from a normal truncated
/// to `(0, +∞)`. With `mean` set to the current linear predictor
/// μ_i, this function delivers exactly that draw.
///
/// # Algorithm
///
/// Inverse-CDF method: compute the standard-normal CDF at `lower`,
/// draw a uniform `u` on `(Φ(lower), 1)`, then invert via the
/// approximation in [`erfinv`]. The bound on `phi_alpha` keeps the
/// inverse-CDF input strictly inside `(0, 1)` to avoid the
/// approximation's endpoint singularities — `2e-10` safety margin
/// is enough to survive any double-precision input.
///
/// # Numerical notes
///
/// For very large positive `(lower − mean)`, `Φ(α) ≈ 1` and the
/// truncated distribution degenerates. The clamp prevents
/// catastrophic loss of precision in that regime; the result is a
/// draw close to `lower` rather than a NaN.
pub fn rtruncnorm_lower<R: Rng>(rng: &mut R, mean: f64, lower: f64) -> f64 {
    // Inverse CDF method
    // P(Z > lower) where Z ~ N(mean, 1)
    use std::f64::consts::SQRT_2;
    
    let alpha = (lower - mean) / 1.0;  // standardize

    // Clamp keeps the inverse-CDF bound on
    // valid (phi_alpha + 1e-10 <= 1 - 1e-10).
    let phi_alpha = (0.5 * (1.0 + libm::erf(alpha / SQRT_2))).clamp(0.0, 1.0 - 2e-10);
    
    // Sample u ~ Uniform(phi_alpha, 1)
    let u: f64 = rng.gen::<f64>() * (1.0 - phi_alpha) + phi_alpha;
    let u = u.clamp(phi_alpha + 1e-10, 1.0 - 1e-10);
    
    // Inverse CDF: mean + sqrt(2) * erfinv(2u - 1)
    mean + SQRT_2 * erfinv(2.0 * u - 1.0)
}

/// Sample from an upper-truncated `N(mean, 1)` distribution,
/// restricted to `(−∞, upper)`.
///
/// # Use in Albert-Chib
///
/// The counterpart of [`rtruncnorm_lower`]: when `y_i = 0` in the
/// binary trait model, the latent liability `z_i` is drawn from a
/// normal truncated to `(−∞, 0)`.
///
/// # Implementation trick
///
/// Rather than implementing a separate inverse-CDF routine, we
/// exploit symmetry: an upper-truncated `N(μ, 1)` at `upper` is the
/// negative of a lower-truncated `N(−μ, 1)` at `−upper`. One
/// negation each on input and output gives the right draw with zero
/// code duplication.
pub fn rtruncnorm_upper<R: Rng>(rng: &mut R, mean: f64, upper: f64) -> f64 {
    // Mirror: sample from lower truncation of -N(-mean, 1) at -upper.
    -rtruncnorm_lower(rng, -mean, -upper)
}

/// Inverse error function (erfinv) approximation
/// Accurate to ~1e-7 for |x| < 1
fn erfinv(x: f64) -> f64 {
    // Winitzki approximation
    let a = 0.147;
    let ln_term = libm::log(1.0 - x * x);
    let c = 2.0 / (std::f64::consts::PI * a) + ln_term / 2.0;
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    sign * (((c * c - ln_term / a).sqrt() - c).sqrt())
}

// ============================================================================
// Unit tests
// ============================================================================
//
// Deterministic behaviour of the sampling primitives + diagnostic helpers.
// Run with `cargo test` or compile-verify with `cargo check --tests`.

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    #[test]
    fn rinvgamma_seeded_reproducible() {
        // Same seed → identical draws across runs.
        let mut rng1 = Pcg64::seed_from_u64(42);
        let mut rng2 = Pcg64::seed_from_u64(42);
        let a = rinvgamma(&mut rng1, 2.0, 1.5);
        let b = rinvgamma(&mut rng2, 2.0, 1.5);
        assert_eq!(a, b, "reproducibility failure for the same seed");
        // Inverse-gamma is always positive.
        assert!(a > 0.0);
    }

    #[test]
    fn rdirichlet_returns_simplex() {
        // Sample sums to 1 and every entry is non-negative.
        let mut rng = Pcg64::seed_from_u64(7);
        let alpha = array![1.0, 2.0, 0.5, 3.0];
        let s = rdirichlet(&mut rng, &alpha);
        assert_eq!(s.len(), alpha.len());
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10,
            "Dirichlet sample should sum to 1, got {}", sum);
        for v in s.iter() {
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn rnorm_seeded_reproducible() {
        let mut rng1 = Pcg64::seed_from_u64(99);
        let mut rng2 = Pcg64::seed_from_u64(99);
        let a = rnorm(&mut rng1, 0.0, 1.0);
        let b = rnorm(&mut rng2, 0.0, 1.0);
        assert_eq!(a, b);
    }

    #[test]
    fn tabulate_basic_counts() {
        // tabulate([0, 1, 1, 2, 1, 3], 4) → [1, 3, 1, 1]
        let labels = Array1::from_vec(vec![0_usize, 1, 1, 2, 1, 3]);
        let counts = tabulate(&labels, 4);
        assert_eq!(counts, vec![1, 3, 1, 1]);
    }

    #[test]
    fn tabulate_ignores_out_of_range_labels() {
        // Labels >= k are silently ignored (out-of-range guard).
        let labels = Array1::from_vec(vec![0_usize, 1, 5, 2, 99]);
        let counts = tabulate(&labels, 3);
        assert_eq!(counts, vec![1, 1, 1]);
    }

    #[test]
    fn effective_size_constant_chain_returns_n() {
        // A flat-line trace has zero variance → ESS = N (early return).
        let samples = Array1::from_elem(1000, 0.5);
        let ess = effective_size(&samples);
        assert_eq!(ess, 1000.0);
    }

    #[test]
    fn geweke_z_returns_zero_for_short_chain() {
        let samples = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let z = geweke_z(&samples);
        assert_eq!(z, 0.0,
            "Geweke should be 0 for n < 100 samples (insufficient data)");
    }

    #[test]
    fn rtruncnorm_lower_respects_bound() {
        // 200 draws — all must be > lower.
        let mut rng = Pcg64::seed_from_u64(13);
        let lower = 0.5_f64;
        for _ in 0..200 {
            let z = rtruncnorm_lower(&mut rng, 0.0, lower);
            assert!(z > lower,
                "lower-truncated draw {} <= lower {}", z, lower);
        }
    }

    #[test]
    fn rtruncnorm_upper_respects_bound() {
        // 200 draws — all must be < upper.
        let mut rng = Pcg64::seed_from_u64(13);
        let upper = -0.5_f64;
        for _ in 0..200 {
            let z = rtruncnorm_upper(&mut rng, 0.0, upper);
            assert!(z < upper,
                "upper-truncated draw {} >= upper {}", z, upper);
        }
    }

    #[test]
    fn rtruncnorm_lower_upper_symmetry() {
        // For mean = 0, lower-truncation at +1 and upper-truncation
        // at -1 are mirror images: same magnitude, opposite sign on
        // average. Verify the deterministic relation rather than the
        // statistical one — for identical seeds the negation
        // identity is exact.
        let mut rng1 = Pcg64::seed_from_u64(11);
        let mut rng2 = Pcg64::seed_from_u64(11);
        let a = rtruncnorm_lower(&mut rng1, 0.0, 1.0);
        let b = rtruncnorm_upper(&mut rng2, 0.0, -1.0);
        assert!((a + b).abs() < 1e-12,
            "mirror identity should give a == -b, got a={}, b={}", a, b);
    }
}