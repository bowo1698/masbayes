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
    wty: Array1<f64>,
    
    // Dimensions
    n: usize,
    n_alleles: usize,
    
    // Hyperparameters
    pi_vec: Array1<f64>,
    sigma2_vec: Array1<f64>,
    
    // Prior parameters
    a0_e: f64,
    b0_e: f64,
    a0_small: f64,
    b0_small: f64,
    a0_medium: f64,
    b0_medium: f64,
    a0_large: f64,
    b0_large: f64,
    
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
}

impl BayesRRunner {
    pub fn new(
        w: Array2<f64>,
        y: Vec<f64>,
        wtw_diag: Vec<f64>,
        wty: Vec<f64>,
        pi_vec: Vec<f64>,
        sigma2_vec: Vec<f64>,
        sigma2_e_init: f64,
        sigma2_ah: f64,
        a0_e: f64, b0_e: f64,
        a0_small: f64, b0_small: f64,
        a0_medium: f64, b0_medium: f64,
        a0_large: f64, b0_large: f64,
        n_iter: usize,
        n_burn: usize,
        n_thin: usize,
        seed: u64,
        fold_id: i32,
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
        
        Self {
            w,
            y: Array1::from_vec(y),
            wtw_diag: Array1::from_vec(wtw_diag),
            wty: Array1::from_vec(wty),
            n,
            n_alleles,
            pi_vec: Array1::from_vec(pi_vec),
            sigma2_vec: Array1::from_vec(sigma2_vec),
            a0_e, b0_e,
            a0_small, b0_small,
            a0_medium, b0_medium,
            a0_large, b0_large,
            n_iter,
            n_burn,
            n_thin,
            rng,
            beta,
            gamma: Array1::<usize>::zeros(n_alleles),
            sigma2_e: sigma2_e_init,
            fold_id,
        }
    }
    
    pub fn run(&mut self) -> BayesRResults {
        let n_save = (self.n_iter - self.n_burn) / self.n_thin;
        
        // Storage
        let mut beta_samples = Array2::<f64>::zeros((n_save, self.n_alleles));
        let mut gamma_samples = Array2::<f64>::zeros((n_save, self.n_alleles));
        let mut sigma2_e_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_small_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_medium_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_large_samples = Array1::<f64>::zeros(n_save);
        let mut pi_samples = Array2::<f64>::zeros((n_save, 4));
        
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
            // 1. Sample beta and gamma
            let mut fitted = self.w.dot(&self.beta);
            let inv_sigma2_e = 1.0 / self.sigma2_e;
            
            for j in 0..self.n_alleles {
                let beta_old = self.beta[j];
                let l_j = self.wtw_diag[j];
                
                // Compute residual correlation
                let mut residuals_prod = self.wty[j];
                for i in 0..self.n {
                    residuals_prod -= self.w[[i, j]] * fitted[i];
                }
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
                
                // Incremental update of fitted values
                if self.beta[j] != beta_old {
                    let delta = self.beta[j] - beta_old;
                    for i in 0..self.n {
                        fitted[i] += self.w[[i, j]] * delta;
                    }
                }
            }
            
            // 2. Sample variance components
            let residuals = &self.y - &fitted;
            let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
            
            let a_e = self.a0_e + (self.n as f64) / 2.0;
            let b_e = self.b0_e + sse / 2.0;
            self.sigma2_e = rinvgamma(&mut self.rng, a_e, b_e);
            
            // Sample mixture variances
            let n_counts = tabulate(&self.gamma, 4);

            // Compute sum of squares for each component in one pass
            let mut ss_components = [0.0; 4];
            for j in 0..self.n_alleles {
                let comp = self.gamma[j];
                if comp > 0 && comp < 4 {
                    ss_components[comp] += self.beta[j].powi(2);
                }
            }

            // Sample variance for each non-zero component
            let prior_a = [self.a0_small, self.a0_medium, self.a0_large];
            let prior_b = [self.b0_small, self.b0_medium, self.b0_large];

            for k in 1..4 {
                let a_k = prior_a[k-1] + (n_counts[k] as f64) / 2.0;
                let b_k = prior_b[k-1] + ss_components[k] / 2.0;
                self.sigma2_vec[k] = rinvgamma(&mut self.rng, a_k, b_k);
            }
            
            // 3. Sample mixture proportions
            let mut alpha_post = Array1::<f64>::ones(4);
            for k in 0..4 {
                alpha_post[k] += n_counts[k] as f64;
            }
            self.pi_vec = rdirichlet(&mut self.rng, &alpha_post);
            
            // 4. Store samples
            if iter >= self.n_burn && (iter - self.n_burn) % self.n_thin == 0 {
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
                save_idx += 1;
            }
            
            // 5. Monitor convergence
            if iter % 1000 == 0 {
                let mean_beta_abs = self.beta.iter().map(|b| b.abs()).sum::<f64>() / (self.n_alleles as f64);
                let non_zero = self.gamma.iter().filter(|&&g| g != 0).count();
                
                eprintln!(
                    "[Fold {}] Iter {} | Mean|β|={:.4} | σ²e={:.4} | π=({:.2},{:.2},{:.2},{:.2}) | Non-zero={}",
                    self.fold_id, iter, mean_beta_abs, self.sigma2_e,
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
        
        BayesRResults {
            beta_samples,
            gamma_samples,
            sigma2_e_samples,
            sigma2_small_samples,
            sigma2_medium_samples,
            sigma2_large_samples,
            pi_samples,
        }
    }
}
