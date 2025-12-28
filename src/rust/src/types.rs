use ndarray::{Array1, Array2};

/// Results from BayesR MCMC
pub struct BayesRResults {
    pub beta_samples: Array2<f64>,
    pub gamma_samples: Array2<f64>,
    pub sigma2_e_samples: Array1<f64>,
    pub sigma2_small_samples: Array1<f64>,
    pub sigma2_medium_samples: Array1<f64>,
    pub sigma2_large_samples: Array1<f64>,
    pub pi_samples: Array2<f64>,
}

/// Results from BayesA MCMC
pub struct BayesAResults {
    pub beta_samples: Array2<f64>,
    pub sigma2_j_samples: Array2<f64>,
    pub sigma2_e_samples: Array1<f64>,
}