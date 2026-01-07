// bayesr_vb.rs
use ndarray::{Array1, Array2};
use crate::types::BayesRResults;

pub struct BayesRVB {
    // Data
    w: Array2<f64>,
    y: Array1<f64>,
    wtw_diag: Array1<f64>,
    wty: Array1<f64>,
    
    // Dimensions
    n: usize,
    n_alleles: usize,
    
    // Variational parameters
    mu: Array1<f64>,           // E[beta]
    tau: Array1<f64>,          // 1/Var[beta]
    omega: Array2<f64>,        // q(gamma) probabilities (n_alleles x 4)
    
    // Hyperparameters
    pi_vec: Array1<f64>,
    sigma2_vec: Array1<f64>,
    sigma2_e: f64,
    
    // VB parameters
    max_iter: usize,
    tol: f64,
    fold_id: i32,
}

impl BayesRVB {
    pub fn new(
        w: Array2<f64>,
        y: Vec<f64>,
        wtw_diag: Vec<f64>,
        wty: Vec<f64>,
        pi_vec: Vec<f64>,
        sigma2_vec: Vec<f64>,
        sigma2_e_init: f64,
        max_iter: usize,
        tol: f64,
        fold_id: i32,
    ) -> Self {
        let n = w.nrows();
        let n_alleles = w.ncols();
        
        // Initialize variational parameters
        let mu = Array1::<f64>::zeros(n_alleles);
        let tau = Array1::<f64>::from_elem(n_alleles, 1.0);
        let mut omega = Array2::<f64>::zeros((n_alleles, 4));
        for j in 0..n_alleles {
            // Start with most mass on zero component
            omega[[j, 0]] = 0.95;
            omega[[j, 1]] = 0.03;
            omega[[j, 2]] = 0.015;
            omega[[j, 3]] = 0.005;
        }
        let sigma2_e = sigma2_e_init.max(0.01);
        }
        
        Self {
            w,
            y: Array1::from_vec(y),
            wtw_diag: Array1::from_vec(wtw_diag),
            wty: Array1::from_vec(wty),
            n,
            n_alleles,
            mu,
            tau,
            omega,
            pi_vec: Array1::from_vec(pi_vec),
            sigma2_vec: Array1::from_vec(sigma2_vec),
            sigma2_e: sigma2_e_init,
            max_iter,
            tol,
            fold_id,
        }
    }
    
    fn compute_elbo(&self) -> f64 {
        let mut elbo = 0.0;
        
        // Likelihood term
        let fitted = self.w.dot(&self.mu);
        let residuals = &self.y - &fitted;
        let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
        
        let trace_term: f64 = (0..self.n_alleles)
            .map(|j| self.wtw_diag[j] * (self.mu[j].powi(2) + 1.0 / self.tau[j]))
            .sum();
        
        elbo -= 0.5 * (self.n as f64) * self.sigma2_e.ln();
        elbo -= 0.5 / self.sigma2_e * (sse + trace_term - 2.0 * self.wty.dot(&self.mu));
        
        // Prior on beta given gamma
        for j in 0..self.n_alleles {
            for k in 0..4 {
                let omega_jk = self.omega[[j, k]];
                if omega_jk > 1e-10 {
                    if k == 0 {
                        // Zero component
                        elbo += omega_jk * self.pi_vec[k].ln();
                    } else {
                        let sigma2_k = self.sigma2_vec[k];
                        elbo += omega_jk * (
                            self.pi_vec[k].ln() 
                            - 0.5 * sigma2_k.ln()
                            - 0.5 / sigma2_k * (self.mu[j].powi(2) + 1.0 / self.tau[j])
                        );
                    }
                }
            }
        }
        
        // Entropy of q(beta)
        for j in 0..self.n_alleles {
            elbo += 0.5 * (1.0 + (2.0 * std::f64::consts::PI / self.tau[j]).ln());
        }
        
        // Entropy of q(gamma)
        for j in 0..self.n_alleles {
            for k in 0..4 {
                let omega_jk = self.omega[[j, k]];
                if omega_jk > 1e-10 {
                    elbo -= omega_jk * omega_jk.ln();
                }
            }
        }
        
        elbo
    }
    
    pub fn run(&mut self) -> BayesRResults {
        eprintln!("[Fold {}] BayesR-VB started: max_iter={}", self.fold_id, self.max_iter);
        eprintln!("[Fold {}] Hyperparameters: π=[{:.3},{:.3},{:.3},{:.3}]",
                  self.fold_id, self.pi_vec[0], self.pi_vec[1], self.pi_vec[2], self.pi_vec[3]);
        
        let mut elbo_prev = f64::NEG_INFINITY;
        let mut converged = false;
        let mut iter = 0;
        
        // Storage for trajectory
        let mut elbo_trajectory = Vec::new();
        let mut pi_trajectory = Vec::new();
        let mut sigma2_trajectory = Vec::new();
        
        while iter < self.max_iter && !converged {
            // Add temperature annealing
            let temperature = if iter < 50 {
                2.0 - (iter as f64 / 50.0)  // Anneal from 2.0 to 1.0
            } else {
                1.0
            };

            // CAVI updates
            
            // 1. Update q(beta_j, gamma_j)
            let mut fitted = self.w.dot(&self.mu);
            let inv_sigma2_e = 1.0 / self.sigma2_e;
            
            for j in 0..self.n_alleles {
                let l_j = self.wtw_diag[j];
                
                // Compute residual contribution
                let mut residuals_prod = self.wty[j];
                for i in 0..self.n {
                    residuals_prod -= self.w[[i, j]] * fitted[i];
                }
                let mu_old = self.mu[j]; 
                let rhs = residuals_prod + l_j * mu_old; 
                
                // Update omega_jk (variational component probabilities)
                let mut log_omega = [0.0; 4];
                
                for k in 0..4 {
                    if k == 0 {
                        // Zero component
                        log_omega[k] = self.pi_vec[k].ln() * temperature;
                    } else {
                        let sigma2_k = self.sigma2_vec[k];
                        let tau_jk = l_j * inv_sigma2_e + 1.0 / sigma2_k;
                        let mu_jk = rhs * inv_sigma2_e / tau_jk;
                        
                        log_omega[k] = self.pi_vec[k].ln() 
                            - 0.5 * sigma2_k.ln()
                            + 0.5 * tau_jk.ln()
                    }       + 0.5 * mu_jk.powi(2) * tau_jk * temperature; 
                }
                
                // Normalize using log-sum-exp
                let max_log = log_omega.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut sum_exp = 0.0;
                for k in 0..4 {
                    sum_exp += (log_omega[k] - max_log).exp();
                }
                
                for k in 0..4 {
                    self.omega[[j, k]] = (log_omega[k] - max_log).exp() / sum_exp;
                }
                
                // Update mu_j and tau_j
                self.mu[j] = 0.0;
                let mut sum_tau_inv = 0.0;
                
                for k in 1..4 {
                    let sigma2_k = self.sigma2_vec[k];
                    let tau_jk = l_j * inv_sigma2_e + 1.0 / sigma2_k;
                    let mu_jk = rhs * inv_sigma2_e / tau_jk;
                    
                    self.mu[j] += self.omega[[j, k]] * mu_jk;
                    sum_tau_inv += self.omega[[j, k]] / tau_jk;
                }
                
                self.tau[j] = if sum_tau_inv > 1e-10 { 1.0 / sum_tau_inv } else { 1e10 };
                
                // Update fitted values
                for i in 0..self.n {
                    fitted[i] += self.w[[i, j]] * (self.mu[j] - mu_old);
                }
            }
            
            // 2. Update sigma2_e (M-step)
            let residuals = &self.y - &fitted;
            let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
            
            let trace_term: f64 = (0..self.n_alleles)
                .map(|j| self.wtw_diag[j] * (self.mu[j].powi(2) + 1.0 / self.tau[j]))
                .sum();
            
            self.sigma2_e = (sse + trace_term - 2.0 * self.wty.dot(&self.mu)) / (self.n as f64);
            self.sigma2_e = self.sigma2_e.max(1e-6);
            
            // 3. Update mixture parameters (M-step)
            for k in 0..4 {
                let n_k: f64 = (0..self.n_alleles).map(|j| self.omega[[j, k]]).sum();
                // Add stronger Dirichlet prior (α₀ = 10 instead of 1)
                let alpha_k = if k == 0 { 
                    10.0  // Strong prior for zero component
                } else {
                    1.0
                };
                self.pi_vec[k] = (n_k + alpha_k) / (self.n_alleles as f64 + 13.0);  // sum(α) = 13
            }
            
            // Update variance components for non-zero components
            for k in 1..4 {
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                
                for j in 0..self.n_alleles {
                    let weight = self.omega[[j, k]];
                    if weight > 1e-6 {  // Only count meaningful weights
                        numerator += weight * (self.mu[j].powi(2) + 1.0 / self.tau[j]);
                        denominator += weight;
                    }
                }
                
                if denominator > 0.1 { 
                    self.sigma2_vec[k] = numerator / denominator;
                    // Enforce hierarchy with bounds
                    let min_var = if k == 1 { 1e-6 } else if k == 2 { 1e-5 } else { 1e-4 };
                    let max_var = if k == 1 { 0.001 } else if k == 2 { 0.01 } else { 0.1 };
                    self.sigma2_vec[k] = self.sigma2_vec[k].max(min_var).min(max_var);
                }
            }
            
            // Check convergence
            let elbo = self.compute_elbo();
            let elbo_change = (elbo - elbo_prev).abs() / (elbo_prev.abs() + 1e-6);
            
            elbo_trajectory.push(elbo);
            pi_trajectory.push(self.pi_vec.to_vec());
            sigma2_trajectory.push([self.sigma2_e, self.sigma2_vec[1], 
                                   self.sigma2_vec[2], self.sigma2_vec[3]]);
            
            if iter % 10 == 0 {
                let mean_beta_abs = self.mu.iter().map(|b| b.abs()).sum::<f64>() / (self.n_alleles as f64);
                let n_nonzero: f64 = (0..self.n_alleles)
                    .map(|j| 1.0 - self.omega[[j, 0]])
                    .sum();
                
                eprintln!("[Fold {}] Iter {} | ELBO={:.2e} | Δ={:.2e} | Mean|μ|={:.4} | σ²e={:.4} | E[nonzero]={:.0}",
                          self.fold_id, iter, elbo, elbo_change, mean_beta_abs, 
                          self.sigma2_e, n_nonzero);
            }
            
            if elbo_change < self.tol && iter > 10 {
                converged = true;
                eprintln!("[Fold {}] Converged at iteration {}", self.fold_id, iter);
            }
            
            elbo_prev = elbo;
            iter += 1;
        }
        
        if !converged {
            eprintln!("[Fold {}] Max iterations reached", self.fold_id);
        }
        
        // Return point estimates as single "sample"
        let beta_samples = {
            let mut mat = Array2::<f64>::zeros((1, self.n_alleles));
            for j in 0..self.n_alleles {
                mat[[0, j]] = self.mu[j];
            }
            mat
        };
        
        let gamma_samples = {
            let mut mat = Array2::<f64>::zeros((1, self.n_alleles));
            for j in 0..self.n_alleles {
                // Expected component
                let expected_comp: f64 = (0..4)
                    .map(|k| (k as f64) * self.omega[[j, k]])
                    .sum();
                mat[[0, j]] = expected_comp;
            }
            mat
        };
        
        let mut pi_samples = Array2::<f64>::zeros((iter, 4));
        for (i, pi) in pi_trajectory.iter().enumerate() {
            for k in 0..4 {
                pi_samples[[i, k]] = pi[k];
            }
        }
        
        BayesRResults {
            beta_samples,
            gamma_samples,
            sigma2_e_samples: Array1::from_vec(sigma2_trajectory.iter().map(|s| s[0]).collect()),
            sigma2_small_samples: Array1::from_vec(sigma2_trajectory.iter().map(|s| s[1]).collect()),
            sigma2_medium_samples: Array1::from_vec(sigma2_trajectory.iter().map(|s| s[2]).collect()),
            sigma2_large_samples: Array1::from_vec(sigma2_trajectory.iter().map(|s| s[3]).collect()),
            pi_samples,
        }
    }
}