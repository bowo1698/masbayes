use ndarray::{Array1, Array2};
use rand::SeedableRng;
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
    freq_weights: Array1<f64>,
    
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
    fold_id: i32,
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
        allele_freqs: Vec<f64>,
        a0_e: f64,
        b0_e: f64,
        n_iter: usize,
        n_burn: usize,
        n_thin: usize,
        seed: u64,
        fold_id: i32,
    ) -> Self {
        let n = w.nrows();
        let n_alleles = w.ncols();
        
        let rng = Pcg64::seed_from_u64(seed);

        let mut freq_weights = Array1::<f64>::ones(n_alleles);
        for j in 0..n_alleles {
            let p = allele_freqs[j];
            freq_weights[j] = (2.0 * p * (1.0 - p)).sqrt();
        }
        
        Self {
            w,
            y: Array1::from_vec(y),
            wtw_diag: Array1::from_vec(wtw_diag),
            wty: Array1::from_vec(wty),
            n,
            n_alleles,
            nu,
            s_squared,
            freq_weights,
            a0_e,
            b0_e,
            n_iter,
            n_burn,
            n_thin,
            rng,
            beta_a: Array1::<f64>::zeros(n_alleles),
            sigma2_j: Array1::<f64>::from_elem(n_alleles, s_squared),
            sigma2_e_a: sigma2_e_init,
            fold_id,
        }
    }
    
    pub fn run(&mut self) -> BayesAResults {
        let n_save = (self.n_iter - self.n_burn) / self.n_thin;
        
        // Storage
        let mut beta_samples = Array2::<f64>::zeros((n_save, self.n_alleles));
        let mut sigma2_j_samples = Array2::<f64>::zeros((n_save, self.n_alleles));
        let mut sigma2_e_samples = Array1::<f64>::zeros(n_save);
        
        let mut save_idx = 0;
        
        eprintln!("[Fold {}] BayesA MCMC started: {} iterations", self.fold_id, self.n_iter);
        eprintln!("[Fold {}] Hyperparameters: ν = {:.2}, S² = {:.6}", 
                  self.fold_id, self.nu, self.s_squared);
        eprintln!("[Fold {}] σ²_e = {:.6}\n", self.fold_id, self.sigma2_e_a);
        
        // MCMC loop
        for iter in 0..self.n_iter {
            // 1. Sample beta_j
            let mut fitted = self.w.dot(&self.beta_a);
            let inv_sigma2_e = 1.0 / self.sigma2_e_a;
            
            for j in 0..self.n_alleles {
                let l_j = self.wtw_diag[j];
                
                // Compute residual correlation
                let mut residuals_prod = self.wty[j];
                for i in 0..self.n {
                    residuals_prod -= self.w[[i, j]] * fitted[i];
                }
                let rhs = residuals_prod + l_j * self.beta_a[j];
                
                // Posterior distribution
                let sigma2_j_adjusted = self.sigma2_j[j] * self.freq_weights[j];
                let inv_var_post = l_j * inv_sigma2_e + 1.0 / sigma2_j_adjusted;
                let var_post = 1.0 / inv_var_post;
                let mu_post = rhs * inv_sigma2_e * var_post;
                
                let beta_old = self.beta_a[j];
                self.beta_a[j] = rnorm(&mut self.rng, mu_post, var_post.sqrt());
                
                // Incremental update
                if self.beta_a[j] != beta_old {
                    let delta = self.beta_a[j] - beta_old;
                    for i in 0..self.n {
                        fitted[i] += self.w[[i, j]] * delta;
                    }
                }
            }
            
            // 2. Sample sigma2_j
            let shape_j = (self.nu + 1.0) / 2.0;
            for j in 0..self.n_alleles {
                let scale_j = (self.nu * self.s_squared + self.beta_a[j].powi(2)) / 2.0;
                self.sigma2_j[j] = rinvgamma(&mut self.rng, shape_j, scale_j);
            }
            
            // 3. Sample sigma2_e
            let residuals = &self.y - &fitted;
            let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
            
            let a_e = self.a0_e + (self.n as f64) / 2.0;
            let b_e = self.b0_e + sse / 2.0;
            self.sigma2_e_a = rinvgamma(&mut self.rng, a_e, b_e);
            
            // 4. Store samples
            if iter >= self.n_burn && (iter - self.n_burn) % self.n_thin == 0 {
                for j in 0..self.n_alleles {
                    beta_samples[[save_idx, j]] = self.beta_a[j];
                    sigma2_j_samples[[save_idx, j]] = self.sigma2_j[j];
                }
                sigma2_e_samples[save_idx] = self.sigma2_e_a;
                save_idx += 1;
            }
            
            // 5. Monitor convergence
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
        
        BayesAResults {
            beta_samples,
            sigma2_j_samples,
            sigma2_e_samples,
        }
    }
}
