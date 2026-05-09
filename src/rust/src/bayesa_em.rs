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

    max_iter: usize,
    tol: f64,

    beta: Array1<f64>,
    sigma2_j: Array1<f64>,
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

impl BayesAEM {
    pub fn new(
        w: Array2<f64>,
        y: Vec<f64>,
        wtw_diag: Vec<f64>,
        x: Option<Array2<f64>>,
        nu: f64,
        s_squared: f64,
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
            nu,
            s_squared,
            max_iter,
            tol,
            beta: Array1::<f64>::zeros(n_alleles),
            sigma2_j: Array1::<f64>::from_elem(n_alleles, s_squared),
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
    
    pub fn run(&mut self) -> BayesAResults {
        if self.verbose {
            eprintln!("[Fold {}] BayesA EM started: max {} iterations", self.fold_id, self.max_iter);
        }

        let print_interval = (self.max_iter / 50).max(1);
        let mut beta_old = Array1::<f64>::zeros(self.n_alleles);
        
        for iter in 0..self.max_iter {
            // E-step
            let expected_inv_sigma2 = self.compute_expected_inv_variance();
            
            // M-step
            self.m_step(&expected_inv_sigma2);
            
            // Compute beta change (Paper's method)
            let diff = &self.beta - &beta_old;
            let change_sq = diff.dot(&diff);
            let beta_norm_sq = self.beta.dot(&self.beta);
            
            let rel_beta_change = if beta_norm_sq > 1e-20 {
                change_sq / beta_norm_sq
            } else {
                f64::INFINITY
            };
            
            // Adaptive min_iter
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
                    eprintln!("[Fold {}] Converged at iteration {} (β_change={:.2e})",
                            self.fold_id, iter, rel_beta_change);
                }
                break;
            }

            if self.verbose && iter % print_interval == 0 {
                let mean_beta = self.beta.iter().map(|b| b.abs()).sum::<f64>() / (self.n_alleles as f64);
                let mean_sigma2_j = self.sigma2_j.iter().sum::<f64>() / (self.n_alleles as f64);
                eprintln!("[Fold {}] Iter {} | β_change={:.2e} | σ²e={:.4} | Mean|β|={:.4} | Mean σ²_j={:.4}",
                        self.fold_id, iter, rel_beta_change,
                        self.sigma2_e, mean_beta, mean_sigma2_j);
            }
            
            beta_old = self.beta.clone();
        }
        
        let beta_samples = Array2::from_shape_fn((1, self.n_alleles), |(_, j)| self.beta[j]);
        let sigma2_j_samples = Array2::from_shape_fn((1, self.n_alleles), |(_, j)| self.sigma2_j[j]);
        
        if self.verbose {
            eprintln!("\n[Fold {}] BayesA EM completed!\n", self.fold_id);
        }

        let beta_hat = self.beta.clone();
        let mu_hat = 0.0;
        let sigma2_e_hat = self.sigma2_e;
        let sigma2_j_hat = self.sigma2_j.clone();
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
        
        BayesAResults {
            beta_samples,
            sigma2_j_samples,
            sigma2_e_samples: Array1::from_vec(vec![self.sigma2_e]),
            mu_samples: Array1::from_vec(vec![0.0]),
            alpha_samples: alpha_samples_out,
            beta_hat,
            mu_hat,
            sigma2_e_hat,
            sigma2_j_hat,
            alpha_hat,
            pred_train,
            sigma2_g,
            h2,
            z_hat: None,
        }
    }
    
    fn compute_expected_inv_variance(&self) -> Array1<f64> {
        let mut expected_inv = Array1::<f64>::zeros(self.n_alleles);
        
        for j in 0..self.n_alleles {
            let a = (self.nu + 1.0) / 2.0;
            let b = (self.nu * self.s_squared + self.beta[j].powi(2)) / 2.0;
            expected_inv[j] = a / b;
        }
        
        expected_inv
    }
    
    fn m_step(&mut self, expected_inv_sigma2: &Array1<f64>) {
        // fitted = W*beta + X*alpha
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