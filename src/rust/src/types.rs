use ndarray::{Array1, Array2};

/// Results from BayesR 
pub struct BayesRResults {
    // Haplotype effects
    pub beta_samples: Array2<f64>,
    pub gamma_samples: Array2<f64>,

    // SNP additive effects (None jika model hap_only)
    pub beta_snp_samples: Option<Array2<f64>>,
    pub gamma_snp_samples: Option<Array2<f64>>,

    //Parameters
    pub sigma2_e_samples: Array1<f64>,
    pub sigma2_small_samples: Array1<f64>,
    pub sigma2_medium_samples: Array1<f64>,
    pub sigma2_large_samples: Array1<f64>,
    pub pi_samples: Array2<f64>,
    pub mu_samples: Array1<f64>,

    // Derived quantities
    pub beta_hat: Array1<f64>,
    pub beta_snp_hat: Option<Array1<f64>>,
    pub mu_hat: f64,
    pub sigma2_e_hat: f64,
    pub gebv_train: Array1<f64>,
    pub sigma2_g: f64,
    pub h2: f64,
}

/// Results from BayesA 
pub struct BayesAResults {
    pub beta_samples: Array2<f64>,
    pub beta_snp_samples: Option<Array2<f64>>,
    pub sigma2_j_samples: Array2<f64>,
    pub sigma2_j_snp_samples: Option<Array2<f64>>,
    pub sigma2_e_samples: Array1<f64>,
    pub mu_samples: Array1<f64>,

    // Derived quantities
    pub beta_hat: Array1<f64>,
    pub beta_snp_hat: Option<Array1<f64>>,
    pub mu_hat: f64,
    pub sigma2_e_hat: f64,
    pub sigma2_j_hat: Array1<f64>,
    pub sigma2_j_snp_hat: Option<Array1<f64>>,
    pub gebv_train: Array1<f64>,
    pub sigma2_g: f64,
    pub h2: f64,
}