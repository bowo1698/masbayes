// src/rust/src/bayesa_em.rs

//! BayesA — stochastic EM variant.
//!
//! Replaces the full Gibbs sweep of [`crate::bayesa`] with an EM-style
//! coordinate-ascent update that uses posterior means (rather than draws)
//! of the per-marker variances $\sigma²_j$. Concretely, instead of
//! sampling $\sigma²_j$ each iteration, the E-step plugs in its
//! conditional expectation and the M-step closes the loop by updating
//! $\beta$ and $\sigma²_e$.
//!
//! ## When to use
//!
//! - Genome-wide screens or large runs where full posterior uncertainty
//!   is not needed and only point estimates of marker effects ($\hat\beta$)
//!   are reported.
//! - Cross-validation folds where many BayesA fits are needed and runtime
//!   is the bottleneck.
//!
//! ## When not to use
//!
//! - Reporting credible intervals, posterior distributions of variance
//!   components, or heritability uncertainty — these require the full
//!   MCMC in [`crate::bayesa`].
//! - Datasets with strong multimodality in the marker-effect posterior;
//!   EM only finds a local mode.
//!
//! ## Output
//!
//! Reuses the [`BayesAResults`] struct so downstream R code does not need
//! to branch on which estimator was used. Sample arrays in the result
//! contain point estimates rather than MCMC traces.

use ndarray::{Array1, Array2};
use crate::types::BayesAResults;

/// Stochastic-EM BayesA runner.
///
/// Holds the full state of a BayesA EM fit between iterations. The
/// fields fall into three groups:
///
/// - **Data**: `w` (design matrix), `y` (response), `wtw_diag`,
///   `wty` (precomputed sufficient statistics).
/// - **Hyperparameters**: `nu`, `s_squared` (scaled inverse-χ² prior
///   on per-marker variance), `max_iter`, `tol`.
/// - **State**: `beta`, `sigma2_j`, `sigma2_e`, plus the optional
///   fixed-effects block (`x`, `alpha`, `xtx_diag`, `xty`).
///
/// Constructed by [`BayesAEM::new`], stepped by [`BayesAEM::run`].
/// After `run()` returns, the struct is consumed — the results
/// struct ([`BayesAResults`]) is the only output.
pub struct BayesAEM {
    /// Design matrix `W`, shape `(n, n_alleles)`.
    w: Array2<f64>,
    /// Response vector `y`, length `n`.
    y: Array1<f64>,
    /// Precomputed diagonal of `W' W`, length `n_alleles`.
    /// Avoids recomputing it inside the per-marker loop.
    wtw_diag: Array1<f64>,
    /// Precomputed `W' y`, length `n_alleles`.
    wty: Array1<f64>,

    /// Number of individuals.
    n: usize,
    /// Number of marker columns.
    n_alleles: usize,

    /// Degrees of freedom of the scaled inverse-χ² prior on `σ²_j`.
    nu: f64,
    /// Scale of the same prior. Larger `s_squared` ⇒ weaker shrinkage.
    s_squared: f64,

    /// Maximum EM iterations.
    max_iter: usize,
    /// Convergence tolerance on the max relative change in `β`.
    tol: f64,

    /// Current marker effects estimate.
    beta: Array1<f64>,
    /// Current per-marker variance estimates.
    sigma2_j: Array1<f64>,
    /// Current residual variance estimate.
    sigma2_e: f64,
    /// Cross-validation fold id (for verbose log prefixes only).
    fold_id: i32,
    /// Verbose flag — `true` ⇒ print per-iteration diagnostics.
    verbose: bool,

    /// Optional fixed-effects design `X`, shape `(n, n_fixed)`.
    x: Option<Array2<f64>>,
    /// Current fixed-effect coefficients, length `n_fixed`.
    alpha: Array1<f64>,
    /// Precomputed diagonal of `X' X`, length `n_fixed`.
    xtx_diag: Array1<f64>,
    /// Precomputed `X' y`, length `n_fixed`.
    xty: Array1<f64>,
    /// Number of fixed-effect columns (`0` if `x = None`).
    n_fixed: usize,
}

impl BayesAEM {
    /// Construct a new EM runner from raw data and hyperparameters.
    ///
    /// Precomputes the sufficient statistics `W' y` and (if `X` is
    /// supplied) `X' y` and `diag(X' X)` once at construction so the
    /// per-iteration loop touches `W` and `X` purely through these
    /// reduced summaries.
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
    
    /// Run the EM loop until convergence or `max_iter`.
    ///
    /// # Algorithm
    ///
    /// Each iteration performs:
    ///
    /// 1. **E-step**: update per-marker variance
    ///    `σ²_j = (ν · S² + β_j²) / (ν + 1)` — the conditional mean of
    ///    the inverse-chi-squared full conditional, replacing the
    ///    Gibbs random draw.
    /// 2. **M-step for β**: coordinate-ascent update of each marker
    ///    effect using the closed-form normal posterior:
    ///    `β_j ← (rhs_j / σ²_e) / (l_j / σ²_e + 1 / σ²_j)`.
    /// 3. **M-step for α** (if fixed effects supplied): normal update
    ///    of each fixed-effect coefficient.
    /// 4. **M-step for σ²_e**: closed-form mode of the inverse-gamma
    ///    full conditional given the current residual sum of squares.
    ///
    /// # Convergence
    ///
    /// Stops when the largest relative change in any `β_j` falls
    /// below `tol`, or when `max_iter` is reached. Unlike Gibbs, EM
    /// is deterministic: identical inputs ⇒ identical output.
    ///
    /// # Returns
    ///
    /// [`BayesAResults`] populated with the final EM point estimates
    /// in place of posterior samples. The R wrapper reuses the same
    /// post-processing as the Gibbs path so downstream code is
    /// algorithm-agnostic.
    pub fn run(&mut self) -> BayesAResults {
        if self.verbose {
            eprintln!("[Fold {}] BayesA EM started: max {} iterations", self.fold_id, self.max_iter);
        }

        let print_interval = (self.max_iter / 50).max(1);
        let mut beta_old = Array1::<f64>::zeros(self.n_alleles);
        
        for iter in 0..self.max_iter {
            // ========================================================
            // E-step — compute the conditional expectation of the
            // marker-precision parameter `1 / σ²_j` given the current
            // marker effect β_j and the prior:
            //
            //     E[1 / σ²_j | β_j] = (ν + 1) / (ν · S² + β_j²).
            //
            // This is the exact mean of the inverse-gamma full
            // conditional that Gibbs would have sampled from. Plugging
            // in the expectation here is what makes BayesA-EM
            // deterministic (vs. Gibbs's stochastic draws).
            // ========================================================
            let expected_inv_sigma2 = self.compute_expected_inv_variance();

            // ========================================================
            // M-step — maximise the expected complete-data log-likelihood
            // w.r.t. β, σ²_e, and (optionally) α. Each is closed-form
            // because the relevant full conditionals are conjugate:
            //
            //     β_j   ← (rhs_j / σ²_e) / (l_j / σ²_e + E[1/σ²_j])
            //     σ²_e  ← (SSE + 2 b₀_e) / (n + 2 a₀_e)
            //     α_k   ← (x_k' yadj + l_k · α_k_old) / l_k         (when X present)
            //
            // Updates are done in coordinate-ascent order; yadj stays
            // in sync incrementally as in the Gibbs path.
            // ========================================================
            self.m_step(&expected_inv_sigma2);

            // Relative β change for convergence — Paper's method:
            //     change = ‖β_new − β_old‖² / ‖β_new‖².
            // Squared norms keep the comparison scale-free; the
            // `beta_norm_sq > 1e-20` guard avoids dividing by zero in
            // the first iteration (when β is still all-zero).
            let diff = &self.beta - &beta_old;
            let change_sq = diff.dot(&diff);
            let beta_norm_sq = self.beta.dot(&self.beta);

            let rel_beta_change = if beta_norm_sq > 1e-20 {
                change_sq / beta_norm_sq
            } else {
                f64::INFINITY
            };

            // Adaptive minimum-iteration floor. Larger samples need
            // more iterations before the convergence test can safely
            // fire — the relative-β-change criterion is noisier at
            // small n_iter. Floor values were tuned empirically on
            // simulated data of various sizes.
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
    
    /// E-step: compute `E[1 / σ²_j | β_j]` for every marker.
    ///
    /// # Derivation
    ///
    /// Given the prior `σ²_j ~ InvChi2(ν, S²)` and a single
    /// observation β_j, the inverse-gamma full conditional has mean
    ///
    /// ```text
    /// E[1 / σ²_j | β_j] = (ν + 1) / (ν · S² + β_j²).
    /// ```
    ///
    /// This is the EM analogue of Gibbs's "draw σ²_j" step: instead
    /// of sampling, we plug in the conditional mean and feed it into
    /// the M-step. The result is deterministic and converges to the
    /// MAP solution rather than to a posterior sample.
    ///
    /// # Performance
    ///
    /// Pure `O(n_alleles)` arithmetic, no memory allocation beyond
    /// the output vector. Called once per EM iteration.
    fn compute_expected_inv_variance(&self) -> Array1<f64> {
        let mut expected_inv = Array1::<f64>::zeros(self.n_alleles);

        for j in 0..self.n_alleles {
            let a = (self.nu + 1.0) / 2.0;
            let b = (self.nu * self.s_squared + self.beta[j].powi(2)) / 2.0;
            expected_inv[j] = a / b;
        }

        expected_inv
    }

    /// M-step: maximise the expected complete-data log-likelihood
    /// given the current `E[1 / σ²_j]` from the E-step.
    ///
    /// # Updates (in coordinate-ascent order)
    ///
    /// 1. **β** — for each marker `j`:
    ///    ```text
    ///    β_j ← (w_j' · resid + l_j · β_j_old) · σ²_e⁻¹
    ///          ÷ (l_j · σ²_e⁻¹ + E[1 / σ²_j])
    ///    ```
    ///    where `resid = y − W β − X α` is the working residual.
    ///    The `+ l_j · β_j_old` term re-injects β_j's own contribution
    ///    into the residual so we condition on (W β − w_j β_j) without
    ///    materialising it. Same trick as the Gibbs path.
    ///
    /// 2. **α** (when fixed effects present) — closed-form normal
    ///    posterior mode per column of X.
    ///
    /// 3. **σ²_e** — closed-form inverse-gamma posterior mode:
    ///    `σ²_e = (SSE + 2 b₀_e) / (n + 2 a₀_e)`.
    ///
    /// The fitted vector is recomputed from scratch at the start
    /// rather than maintained incrementally because the
    /// per-iteration M-step touches all markers in sequence anyway
    /// (less cache-friendly to track per-marker deltas at this
    /// scale).
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