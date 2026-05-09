use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::Rng;
use rand_pcg::Pcg64;
use crate::utils::{rinvgamma, rdirichlet, rnorm, tabulate};
use crate::types::BayesRResults;
use crate::utils;

pub struct BayesRRunner {
    // Data
    w: Array2<f64>,
    y: Array1<f64>,
    wtw_diag: Array1<f64>,
    
    // Dimensions
    n: usize,
    n_alleles: usize,
    
    // Hyperparameters
    pi_vec: Array1<f64>,
    sigma2_vec: Array1<f64>,
    mu: f64,
    
    // Prior parameters
    a0_e: f64,
    b0_e: f64,
    a0_g: f64,
    b0_g: f64,

    // variance classes [0.0, 0.0001, 0.001, 0.01]
    variance_class: Array1<f64>,
    
    // MCMC parameters
    n_iter: usize,
    n_burn: usize,
    n_thin: usize,
    
    // RNG
    rng: Pcg64,
    
    // Current state
    beta: Array1<f64>,
    gamma: Array1<usize>,
    sigma2_e: f64,
    fold_id: i32,

    // Albert-Chib
    is_binary: bool,
    z: Array1<f64>,        // latent liability

    // Working residual: yadj = response - mu - X*alpha - W*beta (maintained).
    yadj: Array1<f64>,

    // Fixed effects (optional). When `n_fixed = 0` no fixed-effect sampling
    // happens and the model reduces exactly to the no-X case.
    x: Option<Array2<f64>>,
    alpha: Array1<f64>,
    xtx_diag: Array1<f64>,
    n_fixed: usize,
}

impl BayesRRunner {
    pub fn new(
        w: Array2<f64>,
        y: Vec<f64>,
        wtw_diag: Vec<f64>,
        x: Option<Array2<f64>>,
        pi_vec: Vec<f64>,
        variance_class: Vec<f64>,        // e.g. [0.0, 0.0001, 0.001, 0.01]
        sigma2_e_init: f64,
        sigma2_ah: f64,
        a0_e: f64, b0_e: f64,
        a0_g: f64, b0_g: f64,  // single prior for base variance
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
        
        // Initialize beta with small random values
        let init_sd = (sigma2_ah / n_alleles as f64).sqrt();
        let mut beta = Array1::<f64>::zeros(n_alleles);
        let mut init_rng = Pcg64::seed_from_u64(seed);
        for i in 0..n_alleles {
            beta[i] = rnorm(&mut init_rng, 0.0, init_sd);
        }

        // Initial base variance estimate
        let varg_init = sigma2_ah / ((1.0 - pi_vec[0]) * n_alleles as f64);
        let sigma2_vec: Vec<f64> = variance_class.iter().map(|&f| f * varg_init).collect();

        let y_arr = Array1::from_vec(y.clone());

        // Initialize z = y for gaussian, same for binary start
        let z_init = y_arr.clone();

        // Fixed-effects setup. With alpha = 0 init, X*alpha contributes 0 so
        // yadj_init does not depend on X.
        let n_fixed = x.as_ref().map(|m| m.ncols()).unwrap_or(0);
        let alpha = Array1::<f64>::zeros(n_fixed);
        let xtx_diag = if let Some(ref xm) = x {
            Array1::from_vec(
                (0..n_fixed)
                    .map(|k| xm.column(k).iter().map(|v| v * v).sum::<f64>())
                    .collect(),
            )
        } else {
            Array1::<f64>::zeros(0)
        };

        // Initial residual: yadj = response - W*beta - X*alpha - mu
        //                       = y - W*beta_init   (alpha=0, mu=0)
        let yadj_init: Array1<f64> = &y_arr - &w.dot(&beta);

        Self {
            w,
            y: y_arr,
            wtw_diag: Array1::from_vec(wtw_diag),
            n,
            n_alleles,
            pi_vec: Array1::from_vec(pi_vec),
            sigma2_vec: Array1::from_vec(sigma2_vec),
            mu: 0.0,
            a0_e, b0_e,
            a0_g, b0_g,
            variance_class: Array1::from_vec(variance_class),
            n_iter,
            n_burn,
            n_thin,
            rng,
            beta,
            gamma: Array1::<usize>::zeros(n_alleles),
            sigma2_e: sigma2_e_init,
            fold_id,
            is_binary,
            z: z_init,
            yadj: yadj_init,
            x,
            alpha,
            xtx_diag,
            n_fixed,
        }
    }
    
    pub fn run(&mut self) -> BayesRResults {
        let n_save = (self.n_iter - self.n_burn) / self.n_thin;
        let mut mu_samples = Array1::<f64>::zeros(n_save);
        
        // Storage
        let mut beta_samples = Array2::<f64>::zeros((n_save, self.n_alleles));
        let mut gamma_samples = Array2::<f64>::zeros((n_save, self.n_alleles));
        let mut sigma2_e_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_small_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_medium_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_large_samples = Array1::<f64>::zeros(n_save);
        let mut pi_samples = Array2::<f64>::zeros((n_save, 4));
        let mut z_samples: Option<Array2<f64>> = if self.is_binary {
            Some(Array2::<f64>::zeros((n_save, self.n)))
        } else {
            None
        };
        let mut alpha_samples: Option<Array2<f64>> = if self.n_fixed > 0 {
            Some(Array2::<f64>::zeros((n_save, self.n_fixed)))
        } else {
            None
        };

        let mut save_idx = 0;
        
        eprintln!("[Fold {}] BayesR MCMC started: {} iterations", self.fold_id, self.n_iter);
        eprintln!("[Fold {}] Hyperparameters: π = [{:.3}, {:.3}, {:.3}, {:.3}]", 
                  self.fold_id,
                  self.pi_vec[0], self.pi_vec[1], self.pi_vec[2], self.pi_vec[3]);
        eprintln!("[Fold {}] σ² = [{:.6}, {:.6}, {:.6}, {:.6}]",
                  self.fold_id,
                  self.sigma2_vec[0], self.sigma2_vec[1], self.sigma2_vec[2], self.sigma2_vec[3]);
        eprintln!("[Fold {}] σ²_e = {:.6}\n", self.fold_id, self.sigma2_e);
        
        // MCMC loop
        for iter in 0..self.n_iter {

            // Albert-Chib data augmentation (binary only) ──────────
            // Invariant: self.yadj = response - mu - W*beta. We rewrite
            // (W*beta)[i] + mu = response[i] - yadj[i] = z_old - yadj[i]
            // so we don't need a fresh W.dot(&beta) call here.
            if self.is_binary {
                for i in 0..self.n {
                    let z_old = self.z[i];
                    let mu_i = z_old - self.yadj[i];   // (W*beta)[i] + mu
                    self.z[i] = if self.y[i] > 0.5 {
                        utils::rtruncnorm_lower(&mut self.rng, mu_i, 0.0)
                    } else {
                        utils::rtruncnorm_upper(&mut self.rng, mu_i, 0.0)
                    };
                    // Maintain invariant under the new z[i]:
                    // yadj_new = z_new - mu - W*beta - X*alpha
                    //          = (z_new - z_old) + yadj_old
                    self.yadj[i] += self.z[i] - z_old;
                }
                // Fix sigma2_e = 1 for identifiability
                self.sigma2_e = 1.0;
            }

            // Sample fixed-effect coefficients alpha (when X provided).
            // Flat prior; conjugate posterior alpha_k ~ N(rhs/l_k, sigma2_e/l_k).
            if let Some(ref x_mat) = self.x {
                for k in 0..self.n_fixed {
                    let l_k = self.xtx_diag[k];
                    if l_k < 1e-10 { continue; }
                    let alpha_old = self.alpha[k];
                    let x_k = x_mat.column(k);
                    let dot_xy: f64 = x_k.iter().zip(self.yadj.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    let rhs = dot_xy + l_k * alpha_old;
                    let mu_post = rhs / l_k;
                    let var_post = self.sigma2_e / l_k;
                    self.alpha[k] = rnorm(&mut self.rng, mu_post, var_post.sqrt());
                    let delta = self.alpha[k] - alpha_old;
                    if delta != 0.0 {
                        for i in 0..self.n {
                            self.yadj[i] -= x_k[i] * delta;
                        }
                    }
                }
            }

            // Sample intercept. mean(response - W*beta - X*alpha) = mean(yadj) + mu.
            let mu_post = self.yadj.iter().sum::<f64>() / self.n as f64 + self.mu;
            let mu_sd = (self.sigma2_e / self.n as f64).sqrt();
            let mu_old = self.mu;
            self.mu = rnorm(&mut self.rng, mu_post, mu_sd);

            // Update yadj for new mu: yadj -= (mu_new - mu_old).
            let mu_delta = self.mu - mu_old;
            if mu_delta != 0.0 {
                self.yadj.mapv_inplace(|v| v - mu_delta);
            }

            // 1. Sample beta and gamma
            let inv_sigma2_e = 1.0 / self.sigma2_e;

            for j in 0..self.n_alleles {
                let beta_old = self.beta[j];
                let l_j = self.wtw_diag[j];

                // residuals_prod = w_col[j] · yadj
                //   = w_col[j] · (response - mu - W*beta)
                //   = wty_j - w_col[j] · (W*beta + mu)   [equivalent to old code]
                let w_j = self.w.column(j);
                let residuals_prod: f64 = w_j.iter()
                    .zip(self.yadj.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let rhs = residuals_prod + l_j * beta_old;
                
                // Marginalized log-probabilities for each component
                let mut log_probs = [0.0; 4];
                log_probs[0] = self.pi_vec[0].ln(); // Zero component
                
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
                
                // Sample component using log-sum-exp trick
                let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut probs = [0.0; 4];
                let mut sum_probs = 0.0;
                
                for k in 0..4 {
                    probs[k] = (log_probs[k] - max_log).exp();
                    sum_probs += probs[k];
                }
                
                for k in 0..4 {
                    probs[k] /= sum_probs;
                }
                
                // Sample component
                let u: f64 = self.rng.gen();
                let mut cumsum = 0.0;
                let mut new_gamma_idx = 0;
                
                for k in 0..4 {
                    cumsum += probs[k];
                    if u < cumsum {
                        new_gamma_idx = k;
                        break;
                    }
                }
                
                self.gamma[j] = new_gamma_idx;
                
                // Sample beta conditional on component
                let sigma2_k_chosen = self.sigma2_vec[new_gamma_idx];
                
                if sigma2_k_chosen < 1e-10 {
                    self.beta[j] = 0.0;
                } else {
                    let inv_var_post = l_j * inv_sigma2_e + 1.0 / sigma2_k_chosen;
                    let var_post = 1.0 / inv_var_post;
                    let mu_post = rhs * inv_sigma2_e * var_post;
                    
                    self.beta[j] = rnorm(&mut self.rng, mu_post, var_post.sqrt());
                }
                
                // Incremental update of yadj (mirrors fitted += delta * w_col[j]):
                // yadj_new = response - mu - W*beta_new = yadj_old - delta * w_col[j].
                if self.beta[j] != beta_old {
                    let delta = self.beta[j] - beta_old;
                    let w_j = self.w.column(j);
                    for i in 0..self.n {
                        self.yadj[i] -= w_j[i] * delta;
                    }
                }
            }

            // 2. Sample variance components.
            // For gaussian: SSE = sum((y - mu - W*beta)^2) = sum(yadj^2).
            if !self.is_binary {
                let sse: f64 = self.yadj.iter().map(|r| r.powi(2)).sum();
                let a_e = self.a0_e + (self.n as f64) / 2.0;
                let b_e = self.b0_e + sse / 2.0;
                self.sigma2_e = rinvgamma(&mut self.rng, a_e, b_e);
                // sigma2_e stays fixed at 1.0 for binary
            }
            
            // Tabulate component counts
            let n_counts = tabulate(&self.gamma, 4);

            // Pooled base variance update
            let mut varg_sum = 0.0;
            let mut n_nz: usize = 0;
            for j in 0..self.n_alleles {
                let comp = self.gamma[j];
                if comp > 0 {
                    varg_sum += self.beta[j].powi(2) / self.variance_class[comp];
                    n_nz += 1;
                }
            }

            let a_g = self.a0_g + (n_nz as f64) / 2.0;
            let b_g = self.b0_g + varg_sum / 2.0;
            let varg = rinvgamma(&mut self.rng, a_g, b_g);

            // Propagate to all components
            for k in 1..4 {
                self.sigma2_vec[k] = varg * self.variance_class[k];
            }
            
            // 3. Sample mixture proportions
            let mut alpha_post = Array1::<f64>::ones(4);
            for k in 0..4 {
                alpha_post[k] += n_counts[k] as f64;
            }
            self.pi_vec = rdirichlet(&mut self.rng, &alpha_post);
            
            // 4. Store samples
            if iter >= self.n_burn && (iter - self.n_burn) % self.n_thin == 0 {
                mu_samples[save_idx] = self.mu;
                for j in 0..self.n_alleles {
                    beta_samples[[save_idx, j]] = self.beta[j];
                    gamma_samples[[save_idx, j]] = self.gamma[j] as f64;
                }
                sigma2_e_samples[save_idx] = self.sigma2_e;
                sigma2_small_samples[save_idx] = self.sigma2_vec[1];
                sigma2_medium_samples[save_idx] = self.sigma2_vec[2];
                sigma2_large_samples[save_idx] = self.sigma2_vec[3];
                for k in 0..4 {
                    pi_samples[[save_idx, k]] = self.pi_vec[k];
                }
                // Store latent liability if binary
                if let Some(ref mut zs) = z_samples {
                    for i in 0..self.n {
                        zs[[save_idx, i]] = self.z[i];
                    }
                }
                if let Some(ref mut as_arr) = alpha_samples {
                    for k in 0..self.n_fixed {
                        as_arr[[save_idx, k]] = self.alpha[k];
                    }
                }
                save_idx += 1;
            }
            
            // 5. Monitor convergence
            let monitor_interval = (self.n_iter / 10).max(100).min(1000);
            if iter % monitor_interval == 0 {
                let mean_beta_abs = self.beta.iter().map(|b| b.abs()).sum::<f64>() / (self.n_alleles as f64);
                let non_zero = self.gamma.iter().filter(|&&g| g != 0).count();
                
                eprintln!(
                    "[Fold {}] Iter {}/{} | Mean|β|={:.4} | σ²e={:.4} | π=({:.2},{:.2},{:.2},{:.2}) | Non-zero={}",
                    self.fold_id, iter, self.n_iter, mean_beta_abs, self.sigma2_e,
                    self.pi_vec[0], self.pi_vec[1], self.pi_vec[2], self.pi_vec[3],
                    non_zero
                );
            }
        }

        // Calculate diagnostics
        let ess = utils::effective_size(&sigma2_e_samples);
        let geweke = utils::geweke_z(&sigma2_e_samples);
        
        eprintln!("[Fold {}] ESS: {:.0} | Geweke Z: {:.3}", self.fold_id, ess, geweke);
        eprintln!("\n[Fold {}] BayesR MCMC completed!\n", self.fold_id);

        // Posterior means
        let beta_hat = beta_samples.mean_axis(ndarray::Axis(0)).unwrap();
        let mu_hat = mu_samples.mean().unwrap();
        let sigma2_e_hat = sigma2_e_samples.mean().unwrap();
        let alpha_hat: Option<Array1<f64>> = alpha_samples.as_ref()
            .map(|asm| asm.mean_axis(ndarray::Axis(0)).unwrap());

        // GEBV train = W * beta_hat + X * alpha_hat + mu_hat
        let mut pred_train = self.w.dot(&beta_hat);
        if let (Some(x_mat), Some(ah)) = (self.x.as_ref(), alpha_hat.as_ref()) {
            let xa = x_mat.dot(ah);
            for i in 0..self.n {
                pred_train[i] += xa[i];
            }
        }
        pred_train.mapv_inplace(|v| v + mu_hat);

        // Variance of GEBV = sigma2_g
        let gebv_mean = pred_train.mean().unwrap();
        let sigma2_g = pred_train.iter()
            .map(|&g| (g - gebv_mean).powi(2))
            .sum::<f64>() / (self.n as f64 - 1.0);

        let h2 = sigma2_g / (sigma2_g + sigma2_e_hat);
        let z_hat = z_samples.map(|zs| zs.mean_axis(ndarray::Axis(0)).unwrap());

        eprintln!("[Fold {}] σ²_g = {:.6} | h² = {:.4}", self.fold_id, sigma2_g, h2);
        
        BayesRResults {
            beta_samples,
            gamma_samples,
            sigma2_e_samples,
            sigma2_small_samples,
            sigma2_medium_samples,
            sigma2_large_samples,
            pi_samples,
            mu_samples,
            alpha_samples,
            beta_hat,
            mu_hat,
            sigma2_e_hat,
            alpha_hat,
            pred_train,
            sigma2_g,
            h2,
            z_hat,
        }
    }
}
