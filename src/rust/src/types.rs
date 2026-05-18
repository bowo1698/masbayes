//! POD result structs returned by the BayesR and BayesA kernels.
//!
//! These types are written by the MCMC / EM runners in `bayesr.rs`,
//! `bayesa.rs`, `bayesr_em.rs`, `bayesa_em.rs` and then marshalled to R
//! lists by the extendr wrappers in `lib.rs`. Field naming matches the
//! manuscript notation:
//!
//! - `beta_samples` / `beta_hat`: marker (allele) effects.
//! - `mu_samples` / `mu_hat`: intercept (grand mean).
//! - `sigma2_e_*`: residual variance.
//! - `sigma2_j_*` (BayesA) and `sigma2_{small,medium,large}_*` (BayesR):
//!   per-marker / per-component variance components.
//! - `gamma_samples` / `pi_samples` (BayesR only): mixture allocations and
//!   their Dirichlet posterior weights.
//! - `alpha_*`: optional fixed-effect coefficients; `None` when no design
//!   matrix `X` was supplied.
//! - `z_hat`: posterior mean of latent liabilities (Albert–Chib) for
//!   binary traits; `None` for continuous traits.
//! - `sigma2_g`, `h2`: derived quantities — total additive variance and
//!   narrow-sense heritability computed from posterior means.

use ndarray::{Array1, Array2};

/// Results from BayesR
pub struct BayesRResults {
    pub beta_samples: Array2<f64>,
    pub gamma_samples: Array2<f64>,
    pub sigma2_e_samples: Array1<f64>,
    pub sigma2_small_samples: Array1<f64>,
    pub sigma2_medium_samples: Array1<f64>,
    pub sigma2_large_samples: Array1<f64>,
    pub pi_samples: Array2<f64>,
    pub mu_samples: Array1<f64>,
    /// Posterior samples of fixed-effect coefficients, n_save x q.
    /// `None` when no fixed-effects design matrix X was supplied.
    pub alpha_samples: Option<Array2<f64>>,

    // Derived quantities
    pub beta_hat: Array1<f64>,
    pub mu_hat: f64,
    pub sigma2_e_hat: f64,
    /// Posterior mean fixed-effect coefficients (length q). `None` if no X.
    pub alpha_hat: Option<Array1<f64>>,
    pub pred_train: Array1<f64>,
    pub sigma2_g: f64,
    pub h2: f64,
    pub z_hat: Option<Array1<f64>>,
}

/// Results from BayesA
pub struct BayesAResults {
    pub beta_samples: Array2<f64>,
    pub sigma2_j_samples: Array2<f64>,
    pub sigma2_e_samples: Array1<f64>,
    pub mu_samples: Array1<f64>,
    /// Posterior samples of fixed-effect coefficients, n_save x q.
    pub alpha_samples: Option<Array2<f64>>,

    // Derived quantities
    pub beta_hat: Array1<f64>,
    pub mu_hat: f64,
    pub sigma2_e_hat: f64,
    pub sigma2_j_hat: Array1<f64>,
    /// Posterior mean fixed-effect coefficients (length q).
    pub alpha_hat: Option<Array1<f64>>,
    pub pred_train: Array1<f64>,
    pub sigma2_g: f64,
    pub h2: f64,
    pub z_hat: Option<Array1<f64>>,
}