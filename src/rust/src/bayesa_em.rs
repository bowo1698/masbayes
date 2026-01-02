// src/rust/src/bayesa_em.rs

use ndarray::{Array1, Array2};
use crate::types::BayesAResults;

pub struct BayesAEM {
    w: Array2<f64>,
    y: Array1<f64>,
    wtw_diag: Array1<f64>,
    wty: Array1<f64>,
    
    n: usize,
    n_alleles: usize,
    
    nu: f64,
    s_squared: f64,
    freq_weights: Array1<f64>,
    
    max_iter: usize,
    tol: f64,
    
    beta: Array1<f64>,
    sigma2_j: Array1<f64>,
    sigma2_e: f64,
    fold_id: i32,
}

impl BayesAEM {
    pub fn new(
        w: Array2<f64>,
        y: Vec<f64>,
        wtw_diag: Vec<f64>,
        wty: Vec<f64>,
        nu: f64,
        s_squared: f64,
        sigma2_e_init: f64,
        allele_freqs: Vec<f64>,
        max_iter: usize,
        tol: f64,
        fold_id: i32,
    ) -> Self {
        let n = w.nrows();
        let n_alleles = w.ncols();

        let mut freq_weights = Array1::<f64>::ones(n_alleles);
        for j in 0..n_alleles {
            let p = allele_freqs[j];
            freq_weights[j] = (2.0 * p * (1.0 - p)).sqrt();
        }
        
        Self {
            w,
            y: Array1::from_vec(y),
            wtw_diag: Array1::from_vec(wtw_diag),
            wty: Array1::from_vec(wty),
            n,
            n_alleles,
            nu,
            s_squared,
            freq_weights,
            max_iter,
            tol,
            beta: Array1::<f64>::zeros(n_alleles),
            sigma2_j: Array1::<f64>::from_elem(n_alleles, s_squared),
            sigma2_e: sigma2_e_init,
            fold_id,
        }
    }
    
    pub fn run(&mut self) -> BayesAResults {
        eprintln!("[Fold {}] BayesA EM started: max {} iterations", self.fold_id, self.max_iter);

        let print_interval = (self.max_iter / 50).max(1);
        
        let mut loglik_old = f64::NEG_INFINITY;
        
        for iter in 0..self.max_iter {
            // E-step
            let expected_inv_sigma2 = self.compute_expected_inv_variance();
            
            // M-step
            self.m_step(&expected_inv_sigma2);
            
            // Compute log-likelihood
            let fitted = self.w.dot(&self.beta);
            let residuals = &self.y - &fitted;
            let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
            let loglik = -0.5 * (self.n as f64) * (2.0 * std::f64::consts::PI * self.sigma2_e).ln() 
                         - 0.5 * sse / self.sigma2_e;
            
            // Check convergence
            if iter > 0 && (loglik - loglik_old).abs() < self.tol {
                eprintln!("[Fold {}] Converged at iteration {}", self.fold_id, iter);
                break;
            }
            
            loglik_old = loglik;
            
            if iter % print_interval == 0 {
                let mean_beta = self.beta.iter().map(|b| b.abs()).sum::<f64>() / (self.n_alleles as f64);
                eprintln!("[Fold {}] Iter {} | LogLik={:.2} | σ²e={:.4} | Mean|β|={:.4}", 
                         self.fold_id, iter, loglik, self.sigma2_e, mean_beta);
            }
        }
        
        let beta_samples = Array2::from_shape_fn((1, self.n_alleles), |(_, j)| self.beta[j]);
        let sigma2_j_samples = Array2::from_shape_fn((1, self.n_alleles), |(_, j)| self.sigma2_j[j]);
        
        eprintln!("\n[Fold {}] BayesA EM completed!\n", self.fold_id);
        
        BayesAResults {
            beta_samples,
            sigma2_j_samples,
            sigma2_e_samples: Array1::from_vec(vec![self.sigma2_e]),
        }
    }
    
    fn compute_expected_inv_variance(&self) -> Array1<f64> {
        let mut expected_inv = Array1::<f64>::zeros(self.n_alleles);
        
        for j in 0..self.n_alleles {
            let a = (self.nu + 1.0) / 2.0;
            let b = (self.nu * self.s_squared + self.beta[j].powi(2)) / 2.0;
            expected_inv[j] = (a / b) / self.freq_weights[j];
        }
        
        expected_inv
    }
    
    fn m_step(&mut self, expected_inv_sigma2: &Array1<f64>) {
        let mut fitted = self.w.dot(&self.beta);
        let inv_sigma2_e = 1.0 / self.sigma2_e;
        
        // Storage for posterior variances
        let mut var_post_vec = Array1::<f64>::zeros(self.n_alleles);
        
        // Update beta
        for j in 0..self.n_alleles {
            let l_j = self.wtw_diag[j];
            
            let mut residuals_prod = self.wty[j];
            for i in 0..self.n {
                residuals_prod -= self.w[[i, j]] * fitted[i];
            }
            let rhs = residuals_prod + l_j * self.beta[j];
            
            let inv_var_post = l_j * inv_sigma2_e + expected_inv_sigma2[j];
            let var_post = 1.0 / inv_var_post;
            let mu_post = rhs * inv_sigma2_e * var_post;
            
            // Store posterior variance
            var_post_vec[j] = var_post;
            
            let beta_old = self.beta[j];
            self.beta[j] = mu_post;
            
            if self.beta[j] != beta_old {
                let delta = self.beta[j] - beta_old;
                for i in 0..self.n {
                    fitted[i] += self.w[[i, j]] * delta;
                }
            }
        }
        
        // Update sigma2_j (include posterior variance)
        for j in 0..self.n_alleles {
            let a = (self.nu + 1.0) / 2.0;
            // E[β²] = μ² + σ²
            let expected_beta_sq = self.beta[j].powi(2) + var_post_vec[j];
            let b = (self.nu * self.s_squared + expected_beta_sq) / 2.0;
            self.sigma2_j[j] = b / (a - 1.0);
        }
        
        // Update sigma2_e
        let residuals = &self.y - &fitted;
        let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
        self.sigma2_e = sse / (self.n as f64);
    }
}