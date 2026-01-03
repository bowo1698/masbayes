// src/rust/src/bayesr_em.rs

use ndarray::{Array1, Array2};
use crate::types::BayesRResults;

pub struct BayesREM {
    w: Array2<f64>,
    y: Array1<f64>,
    wtw_diag: Array1<f64>,
    wty: Array1<f64>,
    
    n: usize,
    n_alleles: usize,
    
    pi_vec: Array1<f64>,
    sigma2_vec: Array1<f64>,
    
    max_iter: usize,
    tol: f64,
    
    beta: Array1<f64>,
    gamma_prob: Array2<f64>,
    sigma2_e: f64,
    fold_id: i32,
}

impl BayesREM {
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
        
        Self {
            w,
            y: Array1::from_vec(y),
            wtw_diag: Array1::from_vec(wtw_diag),
            wty: Array1::from_vec(wty),
            n,
            n_alleles,
            pi_vec: Array1::from_vec(pi_vec),
            sigma2_vec: Array1::from_vec(sigma2_vec),
            max_iter,
            tol,
            beta: Array1::<f64>::zeros(n_alleles),
            gamma_prob: Array2::<f64>::zeros((n_alleles, 4)),
            sigma2_e: sigma2_e_init,
            fold_id,
        }
    }
    
    pub fn run(&mut self) -> BayesRResults {
        eprintln!("[Fold {}] BayesR EM started: max {} iterations", self.fold_id, self.max_iter);

        let print_interval = (self.max_iter / 50).max(1);
        
        let mut loglik_old = f64::NEG_INFINITY;
        let mut converged_count = 0;
        
        for iter in 0..self.max_iter {
            // E-step
            self.e_step();
            
            // M-step
            self.m_step();
            
            // Compute log-likelihood
            let fitted = self.w.dot(&self.beta);
            let residuals = &self.y - &fitted;
            let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
            let loglik = -0.5 * (self.n as f64) * (2.0 * std::f64::consts::PI * self.sigma2_e).ln() 
                         - 0.5 * sse / self.sigma2_e;
            
            let abs_change = (loglik - loglik_old).abs();
    
            // Adaptive parameters based on dataset size
            let (min_iter, scale_factor, consec_needed) = if self.n > 5000 {
                (200, 50.0, 10)  // Large dataset: n > 5000
            } else if self.n > 1000 {
                (100, 10.0, 5)   // Medium dataset: 1000 < n <= 5000
            } else {
                (50, 1.0, 5)     // Small dataset: n <= 1000
            };
            
            // Scale tol based on dataset size
            let abs_thresh = self.tol * scale_factor;
            let consec_thresh = abs_thresh * 2.0;
            
            // Convergence check
            if iter > min_iter {
                // Criterion 1: Absolute change below adaptive threshold
                if abs_change < abs_thresh {
                    eprintln!("[Fold {}] Converged at iteration {} (Δ={:.2e} < tol×{:.1}={:.2e})", 
                            self.fold_id, iter, abs_change, scale_factor, abs_thresh);
                    //                                                   ^^^^^ FIXED: .0f → .1
                    break;
                }
                
                // Criterion 2: Consecutive small changes
                if abs_change < consec_thresh {
                    converged_count += 1;
                } else {
                    converged_count = 0;
                }
                
                if converged_count >= consec_needed {
                    eprintln!("[Fold {}] Converged at iteration {} ({} consecutive < {:.2e})", 
                            self.fold_id, iter, converged_count, consec_thresh);
                    break;
                }
            }
            
            if iter % print_interval == 0 {
                let non_zero_beta = self.beta.iter().filter(|&&b| b.abs() > 1e-6).count();
                eprintln!("[Fold {}] Iter {} | LogLik={:.2} (Δ={:.2e}, tgt={:.2e}) | σ²e={:.4} | |β|>0: {}", 
                        self.fold_id, iter, loglik, abs_change, abs_thresh, 
                        self.sigma2_e, non_zero_beta);
            }
            
            loglik_old = loglik;
        }
        
        // Convert soft probabilities to "samples" format
        let beta_samples = Array2::from_shape_fn((1, self.n_alleles), |(_, j)| self.beta[j]);

        // MAP assignment: argmax_k P(γⱼ=k)
        let gamma_samples = Array2::from_shape_fn((1, self.n_alleles), |(_, j)| {
            let mut max_k = 0;
            let mut max_prob = self.gamma_prob[[j, 0]];
            for k in 1..4 {
                if self.gamma_prob[[j, k]] > max_prob {
                    max_prob = self.gamma_prob[[j, k]];
                    max_k = k;
                }
            }
            max_k as f64
        });
        
        eprintln!("\n[Fold {}] BayesR EM completed!\n", self.fold_id);
        
        BayesRResults {
            beta_samples,
            gamma_samples,
            sigma2_e_samples: Array1::from_vec(vec![self.sigma2_e]),
            sigma2_small_samples: Array1::from_vec(vec![self.sigma2_vec[1]]),
            sigma2_medium_samples: Array1::from_vec(vec![self.sigma2_vec[2]]),
            sigma2_large_samples: Array1::from_vec(vec![self.sigma2_vec[3]]),
            pi_samples: Array2::from_shape_fn((1, 4), |(_, k)| self.pi_vec[k]),
        }
    }
    
    fn e_step(&mut self) {
        let fitted = self.w.dot(&self.beta);
        let inv_sigma2_e = 1.0 / self.sigma2_e;
        
        for j in 0..self.n_alleles {
            let l_j = self.wtw_diag[j];
            
            // Residual for marker j
            let mut residuals_prod = self.wty[j]; // w_j' y
            for i in 0..self.n {
                residuals_prod -= self.w[[i, j]] * fitted[i]; // w_j' (y - Ŷ)
            }
            let rhs = residuals_prod + l_j * self.beta[j];
            
            let mut log_probs = [0.0; 4];
            log_probs[0] = self.pi_vec[0].ln();
            
            for k in 1..4 {
                let sigma2_k = self.sigma2_vec[k];
                if sigma2_k < 1e-10 {
                    log_probs[k] = f64::NEG_INFINITY;
                    continue;
                }
                
                let ratio_var = sigma2_k * inv_sigma2_e;
                let log_det = (1.0 + l_j * ratio_var).ln();
                let quad_term = (rhs.powi(2) * sigma2_k) / 
                               (self.sigma2_e * (self.sigma2_e + l_j * sigma2_k));
                
                log_probs[k] = self.pi_vec[k].ln() - 0.5 * log_det + 0.5 * quad_term;
            }
            
            let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut probs = [0.0; 4];
            let mut sum_probs = 0.0;
            
            for k in 0..4 {
                probs[k] = (log_probs[k] - max_log).exp();
                sum_probs += probs[k];
            }
            
            for k in 0..4 {
                self.gamma_prob[[j, k]] = probs[k] / sum_probs;
            }
        }
    }
    
    fn m_step(&mut self) {
        let mut fitted = self.w.dot(&self.beta);
        let inv_sigma2_e = 1.0 / self.sigma2_e;
        
        // Update beta using mixture of components
        for j in 0..self.n_alleles {
            let l_j = self.wtw_diag[j];
            
            let mut residuals_prod = self.wty[j];
            for i in 0..self.n {
                residuals_prod -= self.w[[i, j]] * fitted[i];
            }
            let rhs = residuals_prod + l_j * self.beta[j];
            
            // Compute mixture posterior: E[β] = Σₖ P(γ=k) E[β|γ=k]
            let mut beta_new = 0.0;
            
            for k in 0..4 {
                let prob_k = self.gamma_prob[[j, k]];
                
                if k == 0 {
                    // Component 0: β = 0
                    beta_new += prob_k * 0.0;
                } else {
                    let sigma2_k = self.sigma2_vec[k];
                    if sigma2_k > 1e-10 {
                        let inv_var_post = l_j * inv_sigma2_e + 1.0 / sigma2_k;
                        let var_post = 1.0 / inv_var_post;
                        let mu_post = rhs * inv_sigma2_e * var_post;
                        
                        beta_new += prob_k * mu_post;
                    }
                }
            }
            
            let beta_old = self.beta[j];
            self.beta[j] = beta_new;
            
            // Update fitted values
            if self.beta[j] != beta_old {
                let delta = self.beta[j] - beta_old;
                for i in 0..self.n {
                    fitted[i] += self.w[[i, j]] * delta;
                }
            }
        }
        
        // Update sigma2_e
        let residuals = &self.y - &fitted;
        let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
        self.sigma2_e = sse / (self.n as f64);
        
        // 3. Update sigma2_k
        for k in 1..4 {
            let mut ss = 0.0;
            let mut n_k_soft = 0.0;
            
            for j in 0..self.n_alleles {
                let prob_k = self.gamma_prob[[j, k]];
                if prob_k < 1e-8 { continue; }
                
                let l_j = self.wtw_diag[j];
                let sigma2_k = self.sigma2_vec[k];
                if sigma2_k < 1e-10 { continue; }
                
                // Recompute posterior for component k
                let mut residuals_prod = self.wty[j];
                for i in 0..self.n {
                    residuals_prod -= self.w[[i, j]] * fitted[i];
                }
                let rhs = residuals_prod + l_j * self.beta[j];
                
                let var_post_k = 1.0 / (l_j / self.sigma2_e + 1.0 / sigma2_k);
                let mu_post_k = rhs / self.sigma2_e * var_post_k;
                
                // E[β²] = μ² + σ²
                ss += prob_k * (mu_post_k.powi(2) + var_post_k);
                n_k_soft += prob_k;
            }
            
            if n_k_soft > 0.1 {
                self.sigma2_vec[k] = (ss / n_k_soft).max(1e-6);
            }
        }
        
        // 4. Update pi
        for k in 0..4 {
            let sum_prob: f64 = (0..self.n_alleles)
                .map(|j| self.gamma_prob[[j, k]])
                .sum();
            self.pi_vec[k] = sum_prob / (self.n_alleles as f64);
        }
    }
}