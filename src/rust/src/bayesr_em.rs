// src/rust/src/bayesr_em.rs

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::Rng;
use rand_pcg::Pcg64;
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
    freq_weights: Array1<f64>, 
    
    max_iter: usize,
    tol: f64,
    
    beta: Array1<f64>,
    gamma_prob: Array2<f64>,
    gamma: Array1<usize>, 
    sigma2_e: f64,
    rng: Pcg64,  
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
        allele_freqs: Vec<f64>,
        use_adaptive_grid: bool, 
        max_iter: usize,
        tol: f64,
        seed: u64,
        fold_id: i32,
    ) -> Self {
        let n = w.nrows();
        let n_alleles = w.ncols();
        let mut freq_weights = Array1::<f64>::ones(n_alleles);
        for j in 0..n_alleles {
            let p = allele_freqs[j];
            let weight = (2.0 * p * (1.0 - p)).sqrt();
            // Clip minimum to avoid extreme shrinkage
            freq_weights[j] = weight.max(0.3);  // minimum 30% weight
        }

        // Adaptive variance grid
        let mut sigma2_vec_array = Array1::from_vec(sigma2_vec);
        if use_adaptive_grid {
            use crate::variance_tuning::{estimate_genetic_variance, 
                                         estimate_architecture_type,
                                         adaptive_variance_grid,
                                         adaptive_variance_grid_sparse};
            
            let y_array = Array1::from_vec(y.clone());
            let sigma2_g = estimate_genetic_variance(&w, &y_array);
            
            let is_sparse = estimate_architecture_type(&w, &y_array);
            
            let variance_grid = if is_sparse {
                eprintln!("[Fold {}] EM: Detected SPARSE architecture → using aggressive grid", fold_id);
                adaptive_variance_grid_sparse(sigma2_g)
            } else {
                eprintln!("[Fold {}] EM: Detected POLYGENIC architecture → using standard grid", fold_id);
                adaptive_variance_grid(sigma2_g)
            };
            
            sigma2_vec_array = Array1::from_vec(variance_grid.to_vec());
            
            eprintln!("[Fold {}] EM: Adaptive variance grid: σ²_g = {:.6}", fold_id, sigma2_g);
            eprintln!("[Fold {}] EM:   [0={:.2e}, small={:.2e}, med={:.2e}, large={:.2e}]",
                     fold_id, variance_grid[0], variance_grid[1], 
                     variance_grid[2], variance_grid[3]);
        }
        
        Self {
            w,
            y: Array1::from_vec(y),
            wtw_diag: Array1::from_vec(wtw_diag),
            wty: Array1::from_vec(wty),
            n,
            n_alleles,
            pi_vec: Array1::from_vec(pi_vec),
            sigma2_vec: sigma2_vec_array,
            freq_weights,
            max_iter,
            tol,
            beta: Array1::<f64>::zeros(n_alleles),
            gamma_prob: Array2::<f64>::zeros((n_alleles, 4)),
            gamma: Array1::<usize>::zeros(n_alleles),
            sigma2_e: sigma2_e_init,
            rng: Pcg64::seed_from_u64(seed),
            fold_id,
        }
    }
    
    pub fn run(&mut self) -> BayesRResults {
        eprintln!("[Fold {}] BayesR EM started: max {} iterations", self.fold_id, self.max_iter);

        let print_interval = (self.max_iter / 50).max(1);
        
        let mut loglik_old = f64::NEG_INFINITY;
        
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
            
            // Check convergence
            if iter > 0 && (loglik - loglik_old).abs() < self.tol {
                eprintln!("[Fold {}] Converged at iteration {}", self.fold_id, iter);
                break;
            }
            
            loglik_old = loglik;
            
            if iter % print_interval == 0 {
                let non_zero = self.gamma.iter().filter(|&&g| g != 0).count();
                eprintln!("[Fold {}] Iter {} | LogLik={:.2} | σ²e={:.4} | Non-zero={}", 
                         self.fold_id, iter, loglik, self.sigma2_e, non_zero);
            }
        }

        // Update gamma using MAP estimate
        for j in 0..self.n_alleles {
            let mut max_prob = 0.0;
            let mut max_k = 0;
            
            for k in 0..4 {
                if self.gamma_prob[[j, k]] > max_prob {
                    max_prob = self.gamma_prob[[j, k]];
                    max_k = k;
                }
            }
            self.gamma[j] = max_k;
        }
        
        // Convert point estimates to "samples" format for compatibility
        let beta_samples = Array2::from_shape_fn((1, self.n_alleles), |(_, j)| self.beta[j]);
        let gamma_samples = Array2::from_shape_fn((1, self.n_alleles), |(_, j)| {
            self.gamma[j] as f64
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
            
            let mut residuals_prod = self.wty[j];
            for i in 0..self.n {
                residuals_prod -= self.w[[i, j]] * fitted[i];
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
                
                let sigma2_j_adjusted = sigma2_k * self.freq_weights[j];
                let ratio_var = sigma2_j_adjusted * inv_sigma2_e;
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
        
        // Storage for posterior variances (needed for variance update)
        let mut var_post_vec = Array1::<f64>::zeros(self.n_alleles);
        
        // Update beta
        for j in 0..self.n_alleles {
            let l_j = self.wtw_diag[j];
            
            let mut residuals_prod = self.wty[j];
            for i in 0..self.n {
                residuals_prod -= self.w[[i, j]] * fitted[i];
            }
            let rhs = residuals_prod + l_j * self.beta[j];
            
            // ← SOFT ASSIGNMENT: weighted average over components
            let mut weighted_mu = 0.0;
            let mut weighted_var = 0.0;
            
            for k in 0..4 {
                let prob_k = self.gamma_prob[[j, k]];
                let sigma2_k = self.sigma2_vec[k];
                
                if k == 0 || sigma2_k < 1e-10 {
                    // Component 0: no contribution to beta
                    continue;
                }
                
                let sigma2_j_adjusted = sigma2_k * self.freq_weights[j];
                let inv_var_post_k = l_j * inv_sigma2_e + 1.0 / sigma2_j_adjusted;
                let var_post_k = 1.0 / inv_var_post_k;
                let mu_post_k = rhs * inv_sigma2_e * var_post_k;
                
                weighted_mu += prob_k * mu_post_k;
                weighted_var += prob_k * (var_post_k + mu_post_k.powi(2));
            }
            
            // Posterior variance: Var[β] = E[β²] - E[β]²
            var_post_vec[j] = weighted_var - weighted_mu.powi(2);

            let beta_old = self.beta[j];
            self.beta[j] = weighted_mu;
            
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
        
        // Update sigma2_k using soft assignments
        for k in 1..4 {
            let mut ss = 0.0;
            let mut total_prob = 0.0;
            
            for j in 0..self.n_alleles {
                let prob_k = self.gamma_prob[[j, k]];
                ss += prob_k * (self.beta[j].powi(2) + var_post_vec[j]);
                total_prob += prob_k;
            }
            
            if total_prob > 1e-6 {
                self.sigma2_vec[k] = ss / total_prob;
            }
        }

        // Update pi using soft assignments
        for k in 0..4 {
            self.pi_vec[k] = self.gamma_prob.column(k).sum() / (self.n_alleles as f64);
        }
    }
}