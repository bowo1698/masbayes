//! BayesR Gibbs sampler.
//!
//! ## Model
//!
//! Marker effects follow a four-component mixture of normals (a spike at
//! zero plus three slab components scaled relative to the total genetic
//! variance):
//!
//! ```text
//! y = 1·μ + X·α + W·β + ε,             ε ~ N(0, σ²_e · I)
//! β_j | γ_j = c ~ N(0, vᶜ · σ²_g),     vᶜ ∈ {0, 1e-4, 1e-3, 1e-2}
//! γ_j        ~ Categorical(π),          γ_j ∈ {1, 2, 3, 4}
//! π          ~ Dirichlet(α₀)
//! σ²_e, σ²_g ~ InvGamma(·, ·)
//! ```
//!
//! Component 1 is the spike (`vᶜ = 0`, exactly zero effect); components 2–4
//! are slabs of increasing variance. Sparsity comes from the high prior
//! mass on the spike under the Dirichlet, while large-effect markers
//! escape into one of the slabs.
//!
//! ## Sampling steps (per Gibbs iteration)
//!
//! 1. For each marker $j$, draw the mixture label $\gamma_j$ from its
//!    full conditional categorical (probabilities proportional to the
//!    marginal likelihood under each component × $\pi_c$).
//! 2. Conditional on $\gamma_j = c$, draw $\beta_j$ from its normal
//!    full conditional (zero when `vᶜ = 0`).
//! 3. Update $\pi$ via Dirichlet–Multinomial conjugacy:
//!    $\pi \mid \gamma \sim \mathrm{Dirichlet}(\alpha_0 + n_c)$ with
//!    $n_c$ tabulated by `tabulate(gamma, 4)`.
//! 4. Update $\sigma²_g$ and $\sigma²_e$ from their inverse-gamma full
//!    conditionals.
//! 5. Update the intercept $\mu$ and (optional) fixed effects $\alpha$.
//! 6. (Binary trait) Albert–Chib augmentation step on latent liabilities.
//!
//! ## State management
//!
//! Identical strategy to BayesA: keep an incremental working residual
//! `yadj` updated after every coordinate move so $W\beta$ is never
//! recomputed from scratch within a sweep.
//!
//! ## Output
//!
//! `BayesRRunner::run` returns [`BayesRResults`] with posterior samples
//! of all parameters, mixture allocations $\gamma$, mixture weights $\pi$,
//! and derived quantities (genetic variance, heritability).

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
    verbose: bool,

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
        verbose: bool,
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
            verbose,
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
        
        if self.verbose {
            eprintln!("[Fold {}] BayesR MCMC started: {} iterations", self.fold_id, self.n_iter);
            eprintln!("[Fold {}] Hyperparameters: π = [{:.3}, {:.3}, {:.3}, {:.3}]",
                      self.fold_id,
                      self.pi_vec[0], self.pi_vec[1], self.pi_vec[2], self.pi_vec[3]);
            eprintln!("[Fold {}] σ² = [{:.6}, {:.6}, {:.6}, {:.6}]",
                      self.fold_id,
                      self.sigma2_vec[0], self.sigma2_vec[1], self.sigma2_vec[2], self.sigma2_vec[3]);
            eprintln!("[Fold {}] σ²_e = {:.6}\n", self.fold_id, self.sigma2_e);
        }
        
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

            // ============================================================
            // Step 4 — Mixture allocation γ_j and marker effect β_j.
            //
            // BayesR posits a four-component mixture on each β_j:
            //     β_j | γ_j = c  ~  N(0, vᶜ · σ²_g),
            //     vᶜ ∈ {0, 1e-4, 1e-3, 1e-2}   (component 1 is the spike).
            // The categorical posterior of γ_j is computed by integrating
            // β_j out under each component (marginalised likelihood):
            //     log p(γ_j = c | rest) ∝ log π_c
            //                              − 0.5 · log(1 + l_j · vᶜ / σ²_e)
            //                              + 0.5 · (rhs²·vᶜ·σ²_g)
            //                                       / (σ²_e (σ²_e + l_j · vᶜ · σ²_g))
            // where rhs = w_j' · yadj + l_j · β_j_old absorbs the current
            // β_j contribution back into the residual (same trick as BayesA).
            //
            // Numerical stability: probabilities are computed via log-sum-exp
            // to avoid underflow when one component dominates.
            //
            // Conditional on γ_j = c (the sampled component):
            //     β_j  ~  N(μ_post, σ²_post)  if vᶜ > 0
            //     β_j  ~  δ(0)                 if vᶜ = 0  (spike)
            //     σ²_post⁻¹ = l_j / σ²_e + 1 / (vᶜ · σ²_g)
            //     μ_post    = σ²_post · rhs / σ²_e
            // yadj is updated incrementally with (β_j_new − β_j_old) · w_j.
            // ============================================================
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

            // ============================================================
            // Step 5 — Residual variance σ²_e (Gibbs, continuous traits).
            //
            //     σ²_e | rest  ~  InvGamma( a₀_e + n/2,  b₀_e + SSE/2 )
            //     SSE = Σ_i yadj_i²  (since yadj = response − μ − W β − X α).
            //
            // Binary traits fix σ²_e = 1 on the liability scale; see Step 1
            // for the probit identifiability constraint.
            // ============================================================
            if !self.is_binary {
                let sse: f64 = self.yadj.iter().map(|r| r.powi(2)).sum();
                let a_e = self.a0_e + (self.n as f64) / 2.0;
                let b_e = self.b0_e + sse / 2.0;
                self.sigma2_e = rinvgamma(&mut self.rng, a_e, b_e);
            }
            
            // ============================================================
            // Step 6 — Genetic-variance scale σ²_g (Gibbs).
            //
            // The three slab components share a common scale σ²_g with
            // component-specific factors v_c (variance_class):
            //     σ²_c = σ²_g · v_c   for c ∈ {1, 2, 3}.
            // Conjugate inverse-gamma update from the non-zero β_j only:
            //     σ²_g | rest  ~  InvGamma( a₀_g + n_nz/2,
            //                                b₀_g + Σ_{j:γ_j>0} β_j²/v_{γ_j} / 2 )
            // where n_nz = #{j : γ_j > 0}. The spike (γ_j = 0) contributes
            // nothing because β_j is exactly zero there.
            // ============================================================
            let n_counts = tabulate(&self.gamma, 4);

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

            for k in 1..4 {
                self.sigma2_vec[k] = varg * self.variance_class[k];
            }

            // ============================================================
            // Step 7 — Mixture proportions π (Dirichlet–Multinomial Gibbs).
            //
            //     π | γ  ~  Dirichlet(α₀ + n_c, c = 1..4)
            // with prior concentration α₀ = 1 (uniform) and n_c the count
            // of markers currently assigned to component c (from `tabulate`
            // above). Sampling π every iteration lets the prior adapt to
            // the observed sparsity in the data — small numbers of large
            // effects pull π toward the slabs; many near-zero effects pull
            // π_1 (the spike) toward 1.
            // ============================================================
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
            if self.verbose && iter % monitor_interval == 0 {
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
        
        if self.verbose {
            eprintln!("[Fold {}] ESS: {:.0} | Geweke Z: {:.3}", self.fold_id, ess, geweke);
            eprintln!("\n[Fold {}] BayesR MCMC completed!\n", self.fold_id);
        }

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

        if self.verbose {
            eprintln!("[Fold {}] σ²_g = {:.6} | h² = {:.4}", self.fold_id, sigma2_g, h2);
        }
        
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
