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
            freq_weights[j] = (2.0 * p * (1.0 - p)).sqrt();
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

            // Stochastic step: sample hard assignments
            self.sample_components();
            
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

    fn sample_components(&mut self) {
        // Stochastic step: sample discrete component assignments
        for j in 0..self.n_alleles {
            let u: f64 = self.rng.gen();
            let mut cumsum = 0.0;
            
            for k in 0..4 {
                cumsum += self.gamma_prob[[j, k]];
                if u < cumsum {
                    self.gamma[j] = k;
                    break;
                }
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
            
            // Use hard assignment from stochastic sampling
            let k = self.gamma[j];
            let sigma2_k = self.sigma2_vec[k];
            
            let inv_var_post = if k == 0 || sigma2_k < 1e-10 {
                l_j * inv_sigma2_e + 1e10
            } else {
                let sigma2_j_adjusted = sigma2_k * self.freq_weights[j];
                l_j * inv_sigma2_e + 1.0 / sigma2_j_adjusted
            };
            
            let var_post = 1.0 / inv_var_post;
            let mu_post = rhs * inv_sigma2_e * var_post;
            
            // Store posterior variance
            var_post_vec[j] = var_post;
            
            let beta_old = self.beta[j];
            self.beta[j] = if self.gamma[j] == 0 {
                0.0
            } else {
                mu_post
            };
            
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
        
        // Update sigma2_k (include posterior variance)
        for k in 1..4 {
            let mut ss = 0.0;
            let mut n_k = 0;
            for j in 0..self.n_alleles {
                if self.gamma[j] == k {
                    // E[β²] = μ² + σ²
                    ss += self.beta[j].powi(2) + var_post_vec[j];
                    n_k += 1;
                }
            }
            if n_k > 0 {
                self.sigma2_vec[k] = ss / (n_k as f64);
            }
        }
        
        // Update pi
        for k in 0..4 {
            let count = self.gamma.iter().filter(|&&g| g == k).count();
            self.pi_vec[k] = (count as f64) / (self.n_alleles as f64);
        }
    }
}