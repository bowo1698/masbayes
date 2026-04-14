use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::Rng;
use rand_pcg::Pcg64;
use crate::utils::{rinvgamma, rnorm};
use crate::types::BayesAResults;
use crate::utils;

pub struct BayesARunner {
    // Data
    w: Array2<f64>,
    y: Array1<f64>,
    wtw_diag: Array1<f64>,
    wty: Array1<f64>,
    
    // Dimensions
    n: usize,
    n_alleles: usize,
    
    // Hyperparameters
    nu: f64,
    s_squared: f64,
    
    // Prior parameters
    a0_e: f64,
    b0_e: f64,
    
    // MCMC parameters
    n_iter: usize,
    n_burn: usize,
    n_thin: usize,
    
    // RNG
    rng: Pcg64,
    
    // Current state
    beta_a: Array1<f64>,
    sigma2_j: Array1<f64>,
    sigma2_e_a: f64,
    mu: f64,
    fold_id: i32,

    // Albert-Chib
    is_binary: bool,
    z: Array1<f64>,
    wtz: Array1<f64>,
}

impl BayesARunner {
    pub fn new(
        w: Array2<f64>,
        y: Vec<f64>,
        wtw_diag: Vec<f64>,
        wty: Vec<f64>,
        nu: f64,
        s_squared: f64,
        sigma2_e_init: f64,
        a0_e: f64,
        b0_e: f64,
        n_iter: usize,
        n_burn: usize,
        n_thin: usize,
        seed: u64,
        fold_id: i32,
        is_binary: bool,
    ) -> Self {
        let n = w.nrows();
        let n_alleles = w.ncols();
        let rng = Pcg64::seed_from_u64(seed);

        let y_arr = Array1::from_vec(y.clone());
        let wty_arr = Array1::from_vec(wty.clone());
        let z_init = y_arr.clone();
        let wtz_init = wty_arr.clone();
        
        Self {
            w,
            y: y_arr,
            wtw_diag: Array1::from_vec(wtw_diag),
            wty: wty_arr,
            n,
            n_alleles,
            nu,
            s_squared,
            a0_e,
            b0_e,
            n_iter,
            n_burn,
            n_thin,
            rng,
            beta_a: Array1::<f64>::zeros(n_alleles),
            sigma2_j: Array1::<f64>::from_elem(n_alleles, s_squared),
            sigma2_e_a: sigma2_e_init,
            mu: 0.0,
            fold_id,
            is_binary,
            z: z_init,
            wtz: wtz_init,
        }
    }
    
    pub fn run(&mut self) -> BayesAResults {
        let n_save = (self.n_iter - self.n_burn) / self.n_thin;
        
        // Storage
        let mut beta_samples = Array2::<f64>::zeros((n_save, self.n_alleles));
        let mut sigma2_j_samples = Array2::<f64>::zeros((n_save, self.n_alleles));
        let mut sigma2_e_samples = Array1::<f64>::zeros(n_save);
        let mut mu_samples = Array1::<f64>::zeros(n_save);
        let mut z_samples: Option<Array2<f64>> = if self.is_binary {
            Some(Array2::<f64>::zeros((n_save, self.n)))
        } else {
            None
        };
        
        let mut save_idx = 0;
        
        eprintln!("[Fold {}] BayesA MCMC started: {} iterations", self.fold_id, self.n_iter);
        eprintln!("[Fold {}] Hyperparameters: ν = {:.2}, S² = {:.6}", 
                  self.fold_id, self.nu, self.s_squared);
        eprintln!("[Fold {}] σ²_e = {:.6}\n", self.fold_id, self.sigma2_e_a);
        
        // MCMC loop
        for iter in 0..self.n_iter {

            // Albert-Chib data augmentation
            if self.is_binary {
                let fitted_z = self.w.dot(&self.beta_a);
                for i in 0..self.n {
                    let mu_i = fitted_z[i] + self.mu;
                    self.z[i] = if self.y[i] > 0.5 {
                        utils::rtruncnorm_lower(&mut self.rng, mu_i, 0.0)
                    } else {
                        utils::rtruncnorm_upper(&mut self.rng, mu_i, 0.0)
                    };
                }
                self.wtz = self.w.t().dot(&self.z);
                self.sigma2_e_a = 1.0;  // fixed for identifiability
            }

            // Sample beta_j — use z if binary
            let response = if self.is_binary { &self.z } else { &self.y };
            let w_beta = self.w.dot(&self.beta_a);

            // Sampling mu dari residual y - W*beta (tanpa mu)
            let resid_sum: f64 = (0..self.n)
                .map(|i| response[i] - w_beta[i])
                .sum();
            let mu_mean = resid_sum / self.n as f64;
            let mu_var = self.sigma2_e_a / self.n as f64;
            self.mu = rnorm(&mut self.rng, mu_mean, mu_var.sqrt());

            // fitted = W*beta + mu_BARU
            let mut fitted = w_beta;
            fitted.mapv_inplace(|v| v + self.mu);
            let inv_sigma2_e = 1.0 / self.sigma2_e_a;
            
            for j in 0..self.n_alleles {
                let l_j = self.wtw_diag[j];
                
                // Sample sigma2_j | beta_j_current 
                let shape_j = (self.nu + 1.0) / 2.0;
                let scale_j = (self.nu * self.s_squared + self.beta_a[j].powi(2)) / 2.0;
                self.sigma2_j[j] = rinvgamma(&mut self.rng, shape_j, scale_j);
                
                // Sample beta_j | sigma2_j_new 
                let wty_j = if self.is_binary { self.wtz[j] } else { self.wty[j] };
                let mut residuals_prod = wty_j;
                for i in 0..self.n {
                    residuals_prod -= self.w[[i, j]] * fitted[i];
                }
                let rhs = residuals_prod + l_j * self.beta_a[j];
                
                let inv_var_post = l_j * inv_sigma2_e + 1.0 / self.sigma2_j[j];
                let var_post = 1.0 / inv_var_post;
                let mu_post = rhs * inv_sigma2_e * var_post;
                
                let beta_old = self.beta_a[j];
                self.beta_a[j] = rnorm(&mut self.rng, mu_post, var_post.sqrt());
                
                // Incremental update fitted
                if self.beta_a[j] != beta_old {
                    let delta = self.beta_a[j] - beta_old;
                    for i in 0..self.n {
                        fitted[i] += self.w[[i, j]] * delta;
                    }
                }
            }

            if iter == 0 {
                let sse0: f64 = (0..self.n)
                    .map(|i| (self.y[i] - fitted[i]).powi(2))
                    .sum();
                eprintln!("[Fold {}] SSE iter0: {:.2} | mu: {:.4}", self.fold_id, sse0, self.mu);
            }

            if iter == 1000 {
                let mean_abs_beta = self.beta_a.iter().map(|x| x.abs()).sum::<f64>() / self.n_alleles as f64;
                let fitted_check = self.w.dot(&self.beta_a);
                let response = if self.is_binary { &self.z } else { &self.y };
                let sse_check: f64 = response.iter().zip(fitted_check.iter())
                    .map(|(r, f)| (r - f - self.mu).powi(2))
                    .sum();
                eprintln!("[Fold {}] iter1000: mean|beta|={:.4} | SSE={:.2} | mu={:.4}", 
                    self.fold_id, mean_abs_beta, sse_check, self.mu);
            }
            
            // Sample sigma2_e
            if !self.is_binary {
                let residuals = &self.y - &fitted;
                let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
                let a_e = self.a0_e + (self.n as f64) / 2.0;
                let b_e = self.b0_e + sse / 2.0;
                self.sigma2_e_a = rinvgamma(&mut self.rng, a_e, b_e);
            }
            
            // Store samples
            if iter >= self.n_burn && (iter - self.n_burn) % self.n_thin == 0 {
                for j in 0..self.n_alleles {
                    beta_samples[[save_idx, j]] = self.beta_a[j];
                    sigma2_j_samples[[save_idx, j]] = self.sigma2_j[j];
                }
                sigma2_e_samples[save_idx] = self.sigma2_e_a;
                mu_samples[save_idx] = self.mu;
                // Store latent liability if binary
                if let Some(ref mut zs) = z_samples {
                    for i in 0..self.n {
                        zs[[save_idx, i]] = self.z[i];
                    }
                }
                save_idx += 1;
            }
            
            // Monitor convergence
            let monitor_interval = (self.n_iter / 10).max(100).min(1000);
            if iter % monitor_interval == 0 {
                let mean_beta_abs = self.beta_a.iter().map(|b| b.abs()).sum::<f64>() / (self.n_alleles as f64);
                let mean_sigma2_j = self.sigma2_j.mean().unwrap();
                let min_sigma2_j = self.sigma2_j.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_sigma2_j = self.sigma2_j.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                
                eprintln!(
                    "[Fold {}] Iter {}/{} | Mean|β|={:.4} | σ²e={:.4} | σ²j: {:.2e}-{:.2e} (mean={:.2e})",
                    self.fold_id, iter, self.n_iter, mean_beta_abs, self.sigma2_e_a,
                    min_sigma2_j, max_sigma2_j, mean_sigma2_j
                );
            }
        }

        // Calculate diagnostics
        let ess = utils::effective_size(&sigma2_e_samples);
        let geweke = utils::geweke_z(&sigma2_e_samples);
        
        eprintln!("[Fold {}] ESS: {:.0} | Geweke Z: {:.3}", self.fold_id, ess, geweke);
        eprintln!("\n[Fold {}] BayesA MCMC completed!\n", self.fold_id);

        // Posterior means
        let beta_hat = beta_samples.mean_axis(ndarray::Axis(0)).unwrap();
        let mu_hat = mu_samples.mean().unwrap();
        let sigma2_e_hat = sigma2_e_samples.mean().unwrap();
        let sigma2_j_hat = sigma2_j_samples.mean_axis(ndarray::Axis(0)).unwrap();

        // GEBV train = W * beta_hat + mu_hat
        let mut pred_train = self.w.dot(&beta_hat);
        pred_train.mapv_inplace(|v| v + mu_hat);

        // Variance of GEBV = sigma2_g
        let gebv_mean = pred_train.mean().unwrap();
        let sigma2_g = pred_train.iter()
            .map(|&g| (g - gebv_mean).powi(2))
            .sum::<f64>() / (self.n as f64 - 1.0);

        let h2 = sigma2_g / (sigma2_g + sigma2_e_hat);
        let z_hat = z_samples.map(|zs| zs.mean_axis(ndarray::Axis(0)).unwrap());

        eprintln!("[Fold {}] σ²_g = {:.6} | h² = {:.4}", self.fold_id, sigma2_g, h2);
        
        BayesAResults {
            beta_samples,
            sigma2_j_samples,
            sigma2_e_samples,
            mu_samples,
            beta_hat,
            mu_hat,
            sigma2_e_hat,
            sigma2_j_hat,
            pred_train,
            sigma2_g,
            h2,
            z_hat,
        }
    }
}
