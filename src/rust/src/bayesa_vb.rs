// bayesa_vb.rs
use ndarray::{Array1, Array2};
use crate::types::BayesAResults;

pub struct BayesAVB {
    w: Array2<f64>,
    y: Array1<f64>,
    wtw_diag: Array1<f64>,
    wty: Array1<f64>,
    
    n: usize,
    n_alleles: usize,
    
    // Variational parameters
    mu: Array1<f64>,
    tau: Array1<f64>,          // precision of q(beta)
    a_j: Array1<f64>,          // shape for q(sigma2_j)
    b_j: Array1<f64>,          // rate for q(sigma2_j)
    
    // Hyperparameters
    nu: f64,
    s_squared: f64,
    sigma2_e: f64,
    
    max_iter: usize,
    tol: f64,
    fold_id: i32,
}

impl BayesAVB {
    pub fn new(
        w: Array2<f64>,
        y: Vec<f64>,
        wtw_diag: Vec<f64>,
        wty: Vec<f64>,
        nu: f64,
        s_squared: f64,
        sigma2_e_init: f64,
        max_iter: usize,
        tol: f64,
        fold_id: i32,
    ) -> Self {
        let n = w.nrows();
        let n_alleles = w.ncols();
        
        let mu = Array1::<f64>::zeros(n_alleles);
        let tau = Array1::<f64>::from_elem(n_alleles, 1.0);
        let a_j = Array1::<f64>::from_elem(n_alleles, (nu + 1.0) / 2.0);
        let b_j = Array1::<f64>::from_elem(n_alleles, nu * s_squared / 2.0);
        
        Self {
            w, 
            y: Array1::from_vec(y),
            wtw_diag: Array1::from_vec(wtw_diag),
            wty: Array1::from_vec(wty),
            n, n_alleles,
            mu, tau, a_j, b_j,
            nu, s_squared,
            sigma2_e: sigma2_e_init,
            max_iter, tol, fold_id,
        }
    }
    
    fn compute_elbo(&self) -> f64 {
        let mut elbo = 0.0;
        
        // Likelihood
        let fitted = self.w.dot(&self.mu);
        let residuals = &self.y - &fitted;
        let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
        
        let trace_term: f64 = (0..self.n_alleles)
            .map(|j| self.wtw_diag[j] * (self.mu[j].powi(2) + 1.0 / self.tau[j]))
            .sum();
        
        elbo -= 0.5 * (self.n as f64) * self.sigma2_e.ln();
        elbo -= 0.5 / self.sigma2_e * (sse + trace_term - 2.0 * self.wty.dot(&self.mu));
        
        // Prior on beta
        for j in 0..self.n_alleles {
            let e_inv_sigma2_j = self.a_j[j] / self.b_j[j];
            elbo += 0.5 * (e_inv_sigma2_j.ln() - 2.0 * std::f64::consts::PI.ln());
            elbo -= 0.5 * e_inv_sigma2_j * (self.mu[j].powi(2) + 1.0 / self.tau[j]);
        }
        
        // Prior on sigma2_j
        let shape = self.nu / 2.0;
        let scale = self.nu * self.s_squared / 2.0;
        for j in 0..self.n_alleles {
            elbo += shape * scale.ln() - (shape + 1.0) * (self.b_j[j] / self.a_j[j]).ln();
            elbo -= scale * self.a_j[j] / self.b_j[j];
            elbo -= utils::lgamma(shape);
        }
        
        // Entropy of q(beta)
        for j in 0..self.n_alleles {
            elbo += 0.5 * (1.0 + (2.0 * std::f64::consts::PI / self.tau[j]).ln());
        }
        
        // Entropy of q(sigma2_j)
        for j in 0..self.n_alleles {
            elbo += self.a_j[j] - self.a_j[j].ln() + self.b_j[j].ln() + utils::lgamma(self.a_j[j]);
            elbo += (1.0 - self.a_j[j]) * utils::digamma(self.a_j[j]);
        }
        
        elbo
    }
    
    pub fn run(&mut self) -> BayesAResults {
        eprintln!("[Fold {}] BayesA-VB started: max_iter={}", self.fold_id, self.max_iter);
        
        let mut elbo_prev = f64::NEG_INFINITY;
        let mut iter = 0;
        let mut converged = false;
        
        let mut elbo_trajectory = Vec::new();
        let mut sigma2_e_trajectory = Vec::new();
        
        while iter < self.max_iter && !converged {
            // Update q(beta_j)
            let fitted = self.w.dot(&self.mu);
            let inv_sigma2_e = 1.0 / self.sigma2_e;
            
            for j in 0..self.n_alleles {
                let l_j = self.wtw_diag[j];
                
                let mut residuals_prod = self.wty[j];
                for i in 0..self.n {
                    residuals_prod -= self.w[[i, j]] * fitted[i];
                }
                let rhs = residuals_prod + l_j * self.mu[j];
                
                let e_inv_sigma2_j = self.a_j[j] / self.b_j[j];
                self.tau[j] = l_j * inv_sigma2_e + e_inv_sigma2_j;
                self.mu[j] = rhs * inv_sigma2_e / self.tau[j];
                
                // Update fitted
                let mu_old = fitted[0];
                for i in 0..self.n {
                    fitted[i] += self.w[[i, j]] * (self.mu[j] - mu_old);
                }
            }
            
            // Update q(sigma2_j)
            for j in 0..self.n_alleles {
                self.a_j[j] = (self.nu + 1.0) / 2.0;
                self.b_j[j] = (self.nu * self.s_squared + self.mu[j].powi(2) + 1.0 / self.tau[j]) / 2.0;
            }
            
            // Update sigma2_e
            let residuals = &self.y - &fitted;
            let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
            let trace_term: f64 = (0..self.n_alleles)
                .map(|j| self.wtw_diag[j] * (self.mu[j].powi(2) + 1.0 / self.tau[j]))
                .sum();
            
            self.sigma2_e = (sse + trace_term - 2.0 * self.wty.dot(&self.mu)) / (self.n as f64);
            self.sigma2_e = self.sigma2_e.max(1e-6);
            
            let elbo = self.compute_elbo();
            let elbo_change = (elbo - elbo_prev).abs() / (elbo_prev.abs() + 1e-6);
            
            elbo_trajectory.push(elbo);
            sigma2_e_trajectory.push(self.sigma2_e);
            
            if iter % 10 == 0 {
                let mean_beta = self.mu.iter().map(|b| b.abs()).sum::<f64>() / (self.n_alleles as f64);
                eprintln!("[Fold {}] Iter {} | ELBO={:.2e} | Δ={:.2e} | Mean|μ|={:.4} | σ²e={:.4}",
                          self.fold_id, iter, elbo, elbo_change, mean_beta, self.sigma2_e);
            }
            
            if elbo_change < self.tol && iter > 10 {
                converged = true;
            }
            
            elbo_prev = elbo;
            iter += 1;
        }
        
        let beta_samples = {
            let mut mat = Array2::<f64>::zeros((1, self.n_alleles));
            for j in 0..self.n_alleles {
                mat[[0, j]] = self.mu[j];
            }
            mat
        };
        
        let sigma2_j_samples = {
            let mut mat = Array2::<f64>::zeros((1, self.n_alleles));
            for j in 0..self.n_alleles {
                mat[[0, j]] = self.b_j[j] / self.a_j[j];  // E[sigma2_j]
            }
            mat
        };
        
        BayesAResults {
            beta_samples,
            sigma2_j_samples,
            sigma2_e_samples: Array1::from_vec(sigma2_e_trajectory),
        }
    }
}