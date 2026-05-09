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