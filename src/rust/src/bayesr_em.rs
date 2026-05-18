// src/rust/src/bayesr_em.rs

//! BayesR — stochastic EM variant.
//!
//! Replaces the full Gibbs sweep of [`crate::bayesr`] with an EM-style
//! update where the mixture allocations $\gamma_j$ are replaced by their
//! posterior probabilities and the Dirichlet draws of $\pi$ are replaced
//! by their conditional expectation under the current allocation counts.
//! Marker effects, variance components, and intercept retain closed-form
//! coordinate updates.
//!
//! ## When to use
//!
//! - Same niche as [`crate::bayesa_em`]: large screens where point
//!   estimates suffice and runtime per fit must be small.
//! - Hybrid pipelines that first run EM to warm-start MCMC, then switch
//!   to the full [`crate::bayesr`] sampler for posterior inference.
//!
//! ## When not to use
//!
//! - Reporting mixture-membership uncertainty (which markers belong to
//!   the spike vs. slab) — EM collapses this to soft probabilities at
//!   the mode and loses the underlying Bernoulli/Dirichlet posterior.
//! - Variance-component inference; same caveats as the BayesA EM variant.
//!
//! ## Output
//!
//! Returns a [`BayesRResults`] populated with EM point estimates in the
//! sample fields so the R-side API is identical to the MCMC runner.

use ndarray::{Array1, Array2};
use crate::types::BayesRResults;

/// Stochastic-EM BayesR runner.
///
/// Mirrors [`crate::bayesa_em::BayesAEM`] but for the four-component
/// mixture model of BayesR. The key differences from the Gibbs path
/// in [`crate::bayesr`]:
///
/// - **Soft membership** (`gamma_prob`) instead of hard sampled
///   labels: each marker carries a `(1, 4)` row of mixture-component
///   responsibilities that get re-estimated every iteration.
/// - **`π` from posterior expectation**, not a Dirichlet draw.
/// - **σ²_e from closed-form** mode, not an inverse-gamma draw.
///
/// Result: deterministic point estimates instead of posterior samples.
/// Useful when speed matters more than uncertainty quantification
/// (e.g. inside cross-validation loops).
pub struct BayesREM {
    /// Design matrix `W`, shape `(n, n_alleles)`.
    w: Array2<f64>,
    /// Response vector `y`, length `n`.
    y: Array1<f64>,
    /// Precomputed `diag(W' W)`, length `n_alleles`.
    wtw_diag: Array1<f64>,
    /// Precomputed `W' y`, length `n_alleles`.
    wty: Array1<f64>,

    /// Number of individuals.
    n: usize,
    /// Number of marker columns.
    n_alleles: usize,

    /// Current mixture proportions, length 4 (spike + 3 slabs).
    pi_vec: Array1<f64>,
    /// Component variances, length 4. The spike (index 0) has
    /// variance 0.
    sigma2_vec: Array1<f64>,

    /// Maximum EM iterations.
    max_iter: usize,
    /// Convergence tolerance on max relative change in `β`.
    tol: f64,

    /// Current marker-effect estimates.
    beta: Array1<f64>,
    /// Soft mixture membership probabilities, shape
    /// `(n_alleles, 4)`. Each row sums to 1.
    gamma_prob: Array2<f64>,
    /// Current residual variance estimate.
    sigma2_e: f64,
    /// CV fold id (verbose log prefix only).
    fold_id: i32,
    /// Verbose progress flag.
    verbose: bool,

    /// Optional fixed-effects design.
    x: Option<Array2<f64>>,
    /// Current fixed-effect coefficients.
    alpha: Array1<f64>,
    /// Precomputed `diag(X' X)`.
    xtx_diag: Array1<f64>,
    /// Precomputed `X' y`.
    xty: Array1<f64>,
    /// Number of fixed-effect columns.
    n_fixed: usize,
}

impl BayesREM {
    /// Construct a new EM runner from raw data and initial values.
    ///
    /// Precomputes `W' y` and, when `x` is supplied, `X' y` and
    /// `diag(X' X)`. The mixture state (`pi_vec`, `sigma2_vec`) is
    /// stored as the initial guess; the EM loop will update both as
    /// it runs. `beta` and `gamma_prob` start at zero / uniform and
    /// converge to point estimates during [`Self::run`].
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
    
    /// Run the BayesR EM loop until convergence or `max_iter`.
    ///
    /// # Algorithm
    ///
    /// Each iteration performs two phases:
    ///
    /// 1. **E-step** — compute the four-component mixture
    ///    responsibilities for every marker:
    ///
    ///    ```text
    ///    γ_prob[j, c] ∝ π_c · N(β_j | 0, v_c · σ²_g),    c ∈ {1, 2, 3, 4}.
    ///    ```
    ///
    ///    Normalised per marker so each row of `gamma_prob` sums to 1.
    ///    These soft memberships replace Gibbs's hard categorical
    ///    draws.
    ///
    /// 2. **M-step** — update everything else conditional on the soft
    ///    memberships:
    ///    - Marker effects β_j (weighted average of the conditional
    ///      means under each component, weighted by `γ_prob[j, ·]`).
    ///    - Fixed effects α (closed-form normal updates).
    ///    - Mixture proportions π from the column sums of
    ///      `gamma_prob`.
    ///    - σ²_g and σ²_e from their inverse-gamma modes.
    ///
    /// # Convergence
    ///
    /// Same criterion as BayesA-EM: stop when relative
    /// `‖β_new − β_old‖² / ‖β_new‖² < tol` after at least
    /// `min_iter` iterations (50–200 depending on `n`).
    ///
    /// # Returns
    ///
    /// [`BayesRResults`] populated with MAP-style point estimates
    /// instead of posterior samples. `gamma_samples` is the
    /// per-marker `argmax_c γ_prob[j, c]` — useful for reporting
    /// the "most likely" mixture assignment without exposing the
    /// soft posterior probabilities.
    pub fn run(&mut self) -> BayesRResults {
        if self.verbose {
            eprintln!("[Fold {}] BayesR EM started: max {} iterations", self.fold_id, self.max_iter);
        }

        let print_interval = (self.max_iter / 50).max(1);

        let mut beta_old = Array1::<f64>::zeros(self.n_alleles);

        for iter in 0..self.max_iter {
            // E-step: compute soft mixture responsibilities γ_prob[j, c].
            self.e_step();

            // M-step: update β, α, π, σ²_g, σ²_e given responsibilities.
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
    
    /// E-step: update the soft mixture responsibilities
    /// `γ_prob[j, c] = P(component c | β_j, current params)`.
    ///
    /// # Computation
    ///
    /// For each marker `j` and each component `c`:
    ///
    /// ```text
    /// γ_prob[j, c] ∝ π_c · N(β_j | 0, v_c · σ²_g)
    /// ```
    ///
    /// where the spike component (c = 0) has v_0 = 0 ⇒ a delta at
    /// zero. The Normalising constant is computed via log-sum-exp
    /// for numerical stability — when one component dominates by
    /// several orders of magnitude, computing in linear scale would
    /// underflow at single precision.
    ///
    /// # Output
    ///
    /// Writes into `self.gamma_prob` (shape `(n_alleles, 4)`). Each
    /// row sums to 1 after the normalisation step.
    ///
    /// The fitted vector `W β + X α` is rebuilt at the top of every
    /// E-step rather than maintained incrementally, mirroring the
    /// BayesA-EM M-step. EM iterations touch all markers anyway, so
    /// the cache trade-off favours the simpler full recomputation.
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

    /// M-step: update model parameters given the soft mixture
    /// responsibilities computed by [`Self::e_step`].
    ///
    /// # Updates
    ///
    /// 1. **Marker effects β** — for each marker, a weighted average
    ///    of the conditional means under each non-spike component:
    ///
    ///    ```text
    ///    β_j = Σ_{c>0} γ_prob[j, c] · (rhs_j / σ²_e) / (l_j / σ²_e + 1 / (v_c · σ²_g))
    ///    ```
    ///
    ///    The spike (c = 0) contributes nothing because its conditional
    ///    mean is exactly 0. The working residual is updated
    ///    incrementally as β_j moves.
    ///
    /// 2. **Fixed effects α** — closed-form normal updates per
    ///    column of X (skipped when no fixed effects supplied).
    ///
    /// 3. **Mixture proportions π** — column sums of `gamma_prob`
    ///    normalised by `n_alleles`. This is the M-step analogue of
    ///    the Dirichlet draw in Gibbs.
    ///
    /// 4. **σ²_g** — weighted inverse-gamma mode using soft
    ///    responsibilities as weights, replacing the
    ///    `Σ_{γ_j > 0}` hard count in Gibbs.
    ///
    /// 5. **σ²_e** — closed-form residual variance from SSE.
    ///
    /// All updates are coordinate-ascent on the expected complete-
    /// data log-likelihood, so each strictly improves (or leaves
    /// unchanged) the objective. This is what guarantees EM's
    /// monotone convergence.
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