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
    verbose: bool,

    // Fixed effects (optional)
    x: Option<Array2<f64>>,
    alpha: Array1<f64>,
    xtx_diag: Array1<f64>,
    xty: Array1<f64>,
    n_fixed: usize,
}

impl BayesREM {
    pub fn new(
        w: Array2<f64>,
        y: Vec<f64>,
        wtw_diag: Vec<f64>,
        x: Option<Array2<f64>>,
        pi_vec: Vec<f64>,
        sigma2_vec: Vec<f64>,
        sigma2_e_init: f64,
        max_iter: usize,
        tol: f64,
        fold_id: i32,
        verbose: bool,
    ) -> Self {
        let n = w.nrows();
        let n_alleles = w.ncols();
        let y_arr = Array1::from_vec(y);
        let wty_arr = w.t().dot(&y_arr);

        let n_fixed = x.as_ref().map(|m| m.ncols()).unwrap_or(0);
        let alpha = Array1::<f64>::zeros(n_fixed);
        let (xtx_diag, xty) = if let Some(ref xm) = x {
            let xtx = Array1::from_vec(
                (0..n_fixed)
                    .map(|k| xm.column(k).iter().map(|v| v * v).sum::<f64>())
                    .collect(),
            );
            let xty = xm.t().dot(&y_arr);
            (xtx, xty)
        } else {
            (Array1::<f64>::zeros(0), Array1::<f64>::zeros(0))
        };

        Self {
            w,
            y: y_arr,
            wtw_diag: Array1::from_vec(wtw_diag),
            wty: wty_arr,
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
            verbose,
            x,
            alpha,
            xtx_diag,
            xty,
            n_fixed,
        }
    }
    
    pub fn run(&mut self) -> BayesRResults {
        if self.verbose {
            eprintln!("[Fold {}] BayesR EM started: max {} iterations", self.fold_id, self.max_iter);
        }

        let print_interval = (self.max_iter / 50).max(1);
        
        let mut beta_old = Array1::<f64>::zeros(self.n_alleles);
        
        for iter in 0..self.max_iter {
            // E-step
            self.e_step();
            
            // M-step
            self.m_step();
            
            // Compute beta change
            let diff = &self.beta - &beta_old;
            let change_sq = diff.dot(&diff);
            let beta_norm_sq = self.beta.dot(&self.beta);
            
            // Avoid division by zero
            let rel_beta_change = if beta_norm_sq > 1e-20 {
                change_sq / beta_norm_sq
            } else {
                f64::INFINITY
            };
            
            // Adaptive parameters based on dataset size
            let min_iter = if self.n > 5000 {
                200
            } else if self.n > 1000 {
                100
            } else {
                50
            };
            
            // Convergence check
            if iter > min_iter && rel_beta_change < self.tol {
                if self.verbose {
                    eprintln!("[Fold {}] Converged at iteration {} (β_change={:.2e} < tol={:.2e})",
                            self.fold_id, iter, rel_beta_change, self.tol);
                }
                break;
            }

            if self.verbose && iter % print_interval == 0 {
                let non_zero_beta = self.beta.iter().filter(|&&b| b.abs() > 1e-6).count();
                eprintln!("[Fold {}] Iter {} | β_change={:.2e} (tgt={:.2e}) | σ²e={:.4} | |β|>0: {}",
                        self.fold_id, iter, rel_beta_change, self.tol,
                        self.sigma2_e, non_zero_beta);
            }
            
            beta_old = self.beta.clone();
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
        
        if self.verbose {
            eprintln!("\n[Fold {}] BayesR EM completed!\n", self.fold_id);
        }

        let beta_hat = self.beta.clone();
        let mu_hat = 0.0;
        let sigma2_e_hat = self.sigma2_e;
        let alpha_hat: Option<Array1<f64>> = if self.n_fixed > 0 {
            Some(self.alpha.clone())
        } else {
            None
        };
        let alpha_samples_out: Option<Array2<f64>> = alpha_hat.as_ref()
            .map(|ah| Array2::from_shape_fn((1, self.n_fixed), |(_, k)| ah[k]));

        let mut pred_train = self.w.dot(&beta_hat);
        if let (Some(x_mat), Some(ah)) = (self.x.as_ref(), alpha_hat.as_ref()) {
            let xa = x_mat.dot(ah);
            for i in 0..self.n { pred_train[i] += xa[i]; }
        }
        pred_train.mapv_inplace(|v| v + mu_hat);

        let gebv_mean = pred_train.mean().unwrap();
        let sigma2_g = pred_train.iter()
            .map(|&g| (g - gebv_mean).powi(2))
            .sum::<f64>() / (self.n as f64 - 1.0);
        let h2 = sigma2_g / (sigma2_g + sigma2_e_hat);
        
        BayesRResults {
            beta_samples,
            gamma_samples,
            sigma2_e_samples: Array1::from_vec(vec![self.sigma2_e]),
            sigma2_small_samples: Array1::from_vec(vec![self.sigma2_vec[1]]),
            sigma2_medium_samples: Array1::from_vec(vec![self.sigma2_vec[2]]),
            sigma2_large_samples: Array1::from_vec(vec![self.sigma2_vec[3]]),
            pi_samples: Array2::from_shape_fn((1, 4), |(_, k)| self.pi_vec[k]),
            mu_samples: Array1::from_vec(vec![0.0]),
            alpha_samples: alpha_samples_out,
            beta_hat,
            mu_hat,
            sigma2_e_hat,
            alpha_hat,
            pred_train,
            sigma2_g,
            h2,
            z_hat: None,
        }
    }
    
    fn e_step(&mut self) {
        // fitted = W*beta + X*alpha (X*alpha = 0 when no fixed effects)
        let mut fitted = self.w.dot(&self.beta);
        if let Some(ref x_mat) = self.x {
            let xa = x_mat.dot(&self.alpha);
            for i in 0..self.n { fitted[i] += xa[i]; }
        }
        let residuals = &self.y - &fitted;
        let wt_residuals = self.w.t().dot(&residuals);  // Vectorized once
        
        let inv_sigma2_e = 1.0 / self.sigma2_e;
        
        for j in 0..self.n_alleles {
            let l_j = self.wtw_diag[j];
            let rhs = wt_residuals[j] + l_j * self.beta[j];
            
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
        // Initialise fitted = W*beta + X*alpha
        let mut fitted = self.w.dot(&self.beta);
        if let Some(ref x_mat) = self.x {
            let xa = x_mat.dot(&self.alpha);
            for i in 0..self.n { fitted[i] += xa[i]; }
        }
        let inv_sigma2_e = 1.0 / self.sigma2_e;

        // Update fixed-effect coefficients alpha (closed-form coordinate descent)
        if let Some(ref x_mat) = self.x {
            for k in 0..self.n_fixed {
                let l_k = self.xtx_diag[k];
                if l_k < 1e-10 { continue; }
                let alpha_old = self.alpha[k];
                let x_k = x_mat.column(k);
                let mut residuals_prod = self.xty[k];
                for i in 0..self.n {
                    residuals_prod -= x_k[i] * fitted[i];
                }
                let rhs = residuals_prod + l_k * alpha_old;
                self.alpha[k] = rhs / l_k;
                let delta = self.alpha[k] - alpha_old;
                if delta != 0.0 {
                    for i in 0..self.n {
                        fitted[i] += x_k[i] * delta;
                    }
                }
            }
        }

        // Update beta with coordinate descent
        for j in 0..self.n_alleles {
            let l_j = self.wtw_diag[j];
            let beta_old = self.beta[j];
            
            // Compute W_j' * (y - ŷ_{-j})
            let mut residuals_prod = self.wty[j];
            for i in 0..self.n {
                residuals_prod -= self.w[[i, j]] * fitted[i];
            }
            let rhs = residuals_prod + l_j * beta_old;
            
            // Mixture posterior mean
            let mut beta_new = 0.0;
            for k in 0..4 {
                let prob_k = self.gamma_prob[[j, k]];
                if k == 0 {
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
            
            self.beta[j] = beta_new;
            
            // Incremental update fitted only (unavoidable)
            if self.beta[j] != beta_old {
                let delta = self.beta[j] - beta_old;
                for i in 0..self.n {
                    fitted[i] += self.w[[i, j]] * delta;
                }
            }
        }
        
        // Compute W'r ONCE after all beta updates
        let residuals = &self.y - &fitted;
        let wt_residuals = self.w.t().dot(&residuals);
        
        // Update sigma2_e
        self.sigma2_e = residuals.dot(&residuals) / (self.n as f64);
        
        // Update sigma2_k - vectorized
        for k in 1..4 {
            let mut ss = 0.0;
            let mut n_k_soft = 0.0;
            
            for j in 0..self.n_alleles {
                let prob_k = self.gamma_prob[[j, k]];
                if prob_k < 1e-8 { continue; }
                
                let l_j = self.wtw_diag[j];
                let sigma2_k = self.sigma2_vec[k];
                if sigma2_k < 1e-10 { continue; }
                
                let rhs = wt_residuals[j] + l_j * self.beta[j];
                let var_post_k = 1.0 / (l_j / self.sigma2_e + 1.0 / sigma2_k);
                let mu_post_k = rhs / self.sigma2_e * var_post_k;
                
                ss += prob_k * (mu_post_k.powi(2) + var_post_k);
                n_k_soft += prob_k;
            }
            
            if n_k_soft > 0.1 {
                self.sigma2_vec[k] = (ss / n_k_soft).max(1e-6);
            }
        }
        
        // Update pi
        for k in 0..4 {
            self.pi_vec[k] = (0..self.n_alleles)
                .map(|j| self.gamma_prob[[j, k]])
                .sum::<f64>() / (self.n_alleles as f64);
        }
    }
}