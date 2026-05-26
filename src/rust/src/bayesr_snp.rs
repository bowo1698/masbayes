//! BayesR Gibbs MCMC for biallelic SNP markers.
//!
//! Mixture prior with 4 effect-size classes (zero, small, medium, large).
//! Per-iteration:
//!   yadj  = y - μ - X·α - W·β  (maintained incrementally)
//!   u     = W·β                (maintained incrementally)
//! σ²_g per iter = var(u); reported as mean over saved iter.
//! Binary traits use Albert-Chib augmentation with σ²_e fixed at 1.

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_pcg::Pcg64;

use crate::types::BayesRResults;
use crate::utils::{rinvgamma, rdirichlet, rnorm};
use crate::utils;

pub struct BayesRSNPRunner {
    w: Array2<f64>,
    y: Array1<f64>,
    wtw_diag: Array1<f64>,

    n: usize,
    n_alleles: usize,

    pi_vec: Array1<f64>,
    fold: Array1<f64>,
    n_fold: usize,
    mu: f64,

    a0_e: f64,
    b0_e: f64,
    a0_g: f64,
    b0_g: f64,

    n_iter: usize,
    n_burn: usize,
    n_thin: usize,
    rng: Pcg64,

    beta: Array1<f64>,
    gamma: Array1<usize>,
    sigma2_e: f64,
    varg: f64,
    fold_id: i32,
    verbose: bool,

    is_binary: bool,
    z: Array1<f64>,

    yadj: Array1<f64>,
    u: Array1<f64>,

    x: Option<Array2<f64>>,
    alpha: Array1<f64>,
    xtx_diag: Array1<f64>,
    n_fixed: usize,
}

impl BayesRSNPRunner {
    pub fn new(
        w: Array2<f64>,
        y: Vec<f64>,
        wtw_diag: Vec<f64>,
        x: Option<Array2<f64>>,
        pi_vec: Vec<f64>,
        fold: Vec<f64>,
        sigma2_e_init: f64,
        sigma2_ah: f64,
        a0_e: f64, b0_e: f64,
        a0_g: f64, b0_g: f64,
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
        let n_fold = fold.len();
        let rng = Pcg64::seed_from_u64(seed);

        let varg_init = sigma2_ah / ((1.0 - pi_vec[0]) * n_alleles as f64);

        let y_arr = Array1::from_vec(y);
        let beta = Array1::<f64>::zeros(n_alleles);
        let yadj_init = y_arr.clone();
        let z_init = y_arr.clone();
        let u_init = Array1::<f64>::zeros(n);

        // Binary fixes σ²_e = 1 for probit identifiability.
        let sigma2_e = if is_binary { 1.0 } else { sigma2_e_init };

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

        Self {
            w,
            y: y_arr,
            wtw_diag: Array1::from_vec(wtw_diag),
            n,
            n_alleles,
            pi_vec: Array1::from_vec(pi_vec),
            fold: Array1::from_vec(fold),
            n_fold,
            mu: 0.0,
            a0_e, b0_e,
            a0_g, b0_g,
            n_iter,
            n_burn,
            n_thin,
            rng,
            beta,
            gamma: Array1::<usize>::zeros(n_alleles),
            sigma2_e,
            varg: varg_init,
            fold_id,
            verbose,
            is_binary,
            z: z_init,
            yadj: yadj_init,
            u: u_init,
            x,
            alpha,
            xtx_diag,
            n_fixed,
        }
    }

    pub fn run(&mut self) -> BayesRResults {
        let n_save = (self.n_iter - self.n_burn) / self.n_thin;

        let mut beta_samples = Array2::<f64>::zeros((n_save, self.n_alleles));
        let mut gamma_samples = Array2::<f64>::zeros((n_save, self.n_alleles));
        let mut sigma2_e_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_small_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_medium_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_large_samples = Array1::<f64>::zeros(n_save);
        let mut pi_samples = Array2::<f64>::zeros((n_save, self.n_fold));
        let mut mu_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_g_iter_store = Array1::<f64>::zeros(n_save);

        let mut z_samples: Option<Array2<f64>> = if self.is_binary {
            Some(Array2::<f64>::zeros((n_save, self.n)))
        } else {
            None
        };

        let mut alpha_samples = if self.n_fixed > 0 {
            Some(Array2::<f64>::zeros((n_save, self.n_fixed)))
        } else {
            None
        };

        let mut save_idx = 0usize;

        if self.verbose {
            eprintln!("[Fold {}] BayesR-SNP MCMC | n={} p={} n_iter={} n_burn={} binary={}",
                      self.fold_id, self.n, self.n_alleles,
                      self.n_iter, self.n_burn, self.is_binary);
        }

        for iter in 0..self.n_iter {

            // Albert-Chib latent liability draw for binary y.
            // yadj = response - μ - W·β; mean of liability i = response_i - yadj_i.
            if self.is_binary {
                for i in 0..self.n {
                    let z_old = self.z[i];
                    let mu_i = z_old - self.yadj[i];
                    self.z[i] = if self.y[i] > 0.5 {
                        utils::rtruncnorm_lower(&mut self.rng, mu_i, 0.0)
                    } else {
                        utils::rtruncnorm_upper(&mut self.rng, mu_i, 0.0)
                    };
                    self.yadj[i] += self.z[i] - z_old;
                }
                self.sigma2_e = 1.0;
            }

            // μ ~ N( mean(yadj), σ²_e / n )
            let mean_yadj: f64 = self.yadj.iter().sum::<f64>() / (self.n as f64);
            let mu_step = -rnorm(
                &mut self.rng,
                mean_yadj,
                (self.sigma2_e / self.n as f64).sqrt(),
            );
            self.mu -= mu_step;
            for r in self.yadj.iter_mut() { *r += mu_step; }

            // Fixed effects: per-column conjugate normal.
            if self.n_fixed > 0 {
                if let Some(ref xm) = self.x {
                    for k in 0..self.n_fixed {
                        let xx = self.xtx_diag[k];
                        if xx < 1e-10 { continue; }
                        let old_a = self.alpha[k];
                        let mut rhs: f64 = 0.0;
                        for i in 0..self.n {
                            rhs += xm[[i, k]] * self.yadj[i];
                        }
                        rhs += xx * old_a;
                        let new_a = rnorm(&mut self.rng,
                                          rhs / xx,
                                          (self.sigma2_e / xx).sqrt());
                        let delta = old_a - new_a;
                        for i in 0..self.n {
                            self.yadj[i] += delta * xm[[i, k]];
                        }
                        self.alpha[k] = new_a;
                    }
                }
            }

            // Mixture marker sampling. For each marker:
            //   1. Compute class log-likelihoods s[j] (zero class fixed at log π_0)
            //   2. Softmax → class assignment via cumulative draw
            //   3. β_i ~ N(rhs/v, σ²_e/v) if non-zero class else 0
            //   4. Update yadj, u, varg accumulator
            let mut log_pi = Array1::<f64>::zeros(self.n_fold);
            for j in 0..self.n_fold {
                log_pi[j] = self.pi_vec[j].ln();
            }
            let mut vara_fold = Array1::<f64>::zeros(self.n_fold);
            for j in 0..self.n_fold {
                vara_fold[j] = self.varg * self.fold[j];
            }
            let mut vare_vara_fold = Array1::<f64>::zeros(self.n_fold);
            for j in 1..self.n_fold {
                if vara_fold[j] > 1e-30 {
                    vare_vara_fold[j] = self.sigma2_e / vara_fold[j];
                } else {
                    vare_vara_fold[j] = f64::INFINITY;
                }
            }

            let mut varg_accum: f64 = 0.0;
            let mut snp_class_count = vec![0usize; self.n_fold];

            let mut s = Array1::<f64>::zeros(self.n_fold);
            s[0] = log_pi[0];
            let mut stemp = Array1::<f64>::zeros(self.n_fold);

            for i in 0..self.n_alleles {
                let xx = self.wtw_diag[i];
                if xx < 1e-30 {
                    // Monomorphic marker: skip mixture assignment and Pi
                    // count so Pi[0] posterior reflects only informative
                    // markers (pre-v0.6.2 counted these into class 0).
                    continue;
                }
                let old_gi = self.beta[i];

                let mut rhs: f64 = 0.0;
                for k in 0..self.n {
                    rhs += self.w[[k, i]] * self.yadj[k];
                }
                if old_gi != 0.0 {
                    rhs += xx * old_gi;
                }
                let lhs = xx / self.sigma2_e;

                for j in 1..self.n_fold {
                    let logdetv = (vara_fold[j] * lhs + 1.0).ln();
                    let uhat = rhs / (xx + vare_vara_fold[j]);
                    s[j] = -0.5 * (logdetv - (rhs * uhat / self.sigma2_e)) + log_pi[j];
                }

                for j in 0..self.n_fold {
                    let mut total: f64 = 0.0;
                    for k in 0..self.n_fold {
                        total += (s[k] - s[j]).exp();
                    }
                    stemp[j] = 1.0 / total;
                }

                use rand::Rng;
                let rval: f64 = self.rng.gen::<f64>();

                let mut chosen_class: usize = 0;
                let mut acc: f64 = 0.0;
                for j in 0..self.n_fold {
                    acc += stemp[j];
                    if rval < acc {
                        chosen_class = j;
                        break;
                    }
                }
                self.gamma[i] = chosen_class;
                snp_class_count[chosen_class] += 1;

                if chosen_class > 0 {
                    let v = xx + vare_vara_fold[chosen_class];
                    let gi = rnorm(&mut self.rng,
                                   rhs / v,
                                   (self.sigma2_e / v).sqrt());
                    let delta = old_gi - gi;
                    for k in 0..self.n {
                        let wki = self.w[[k, i]];
                        self.yadj[k] += delta * wki;
                        self.u[k]    -= delta * wki;
                    }
                    self.beta[i] = gi;
                    varg_accum += gi * gi / self.fold[chosen_class];
                } else {
                    let gi = 0.0;
                    if old_gi != 0.0 {
                        let delta = old_gi;
                        for k in 0..self.n {
                            let wki = self.w[[k, i]];
                            self.yadj[k] += delta * wki;
                            self.u[k]    -= delta * wki;
                        }
                    }
                    self.beta[i] = gi;
                }
            }

            // Base variance scale: varg ~ InvGamma( a0_g + n_nz/2, b0_g + Σβ²/fold / 2 )
            let nnz_count: usize = (1..self.n_fold).map(|j| snp_class_count[j]).sum();
            if nnz_count > 0 {
                let shape = (self.a0_g as f64) + (nnz_count as f64) / 2.0;
                let scale = self.b0_g + varg_accum / 2.0;
                self.varg = rinvgamma(&mut self.rng, shape, scale);
            }

            // Pi ~ Dirichlet( 1 + class_counts )
            let mut alpha_post = Array1::<f64>::ones(self.n_fold);
            for j in 0..self.n_fold {
                alpha_post[j] += snp_class_count[j] as f64;
            }
            self.pi_vec = rdirichlet(&mut self.rng, &alpha_post);

            // σ²_e ~ InvGamma( a0_e + n/2, b0_e + SSE/2 ); binary keeps σ²_e = 1.
            if !self.is_binary {
                let sse: f64 = self.yadj.iter().map(|r| r * r).sum();
                let a_e = self.a0_e + (self.n as f64) / 2.0;
                let b_e = self.b0_e + sse / 2.0;
                self.sigma2_e = rinvgamma(&mut self.rng, a_e, b_e);
            }

            // σ²_g iter = var(u)
            let u_mean: f64 = self.u.iter().sum::<f64>() / (self.n as f64);
            let sigma2_g_iter: f64 = self.u.iter()
                .map(|&g| (g - u_mean).powi(2))
                .sum::<f64>() / ((self.n as f64) - 1.0);

            if iter >= self.n_burn && (iter - self.n_burn) % self.n_thin == 0
               && save_idx < n_save {
                mu_samples[save_idx] = self.mu;
                for j in 0..self.n_alleles {
                    beta_samples[[save_idx, j]] = self.beta[j];
                    gamma_samples[[save_idx, j]] = self.gamma[j] as f64;
                }
                sigma2_e_samples[save_idx] = self.sigma2_e;
                if self.n_fold >= 4 {
                    sigma2_small_samples[save_idx]  = self.varg * self.fold[1];
                    sigma2_medium_samples[save_idx] = self.varg * self.fold[2];
                    sigma2_large_samples[save_idx]  = self.varg * self.fold[3];
                }
                for j in 0..self.n_fold {
                    pi_samples[[save_idx, j]] = self.pi_vec[j];
                }
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
                sigma2_g_iter_store[save_idx] = sigma2_g_iter;
                save_idx += 1;
            }
        }

        let beta_hat = beta_samples.mean_axis(ndarray::Axis(0)).unwrap();
        let mu_hat = mu_samples.mean().unwrap();
        let sigma2_e_hat = sigma2_e_samples.mean().unwrap();
        let alpha_hat: Option<Array1<f64>> = alpha_samples.as_ref()
            .map(|asm| asm.mean_axis(ndarray::Axis(0)).unwrap());

        let mut pred_train = self.w.dot(&beta_hat);
        if let (Some(x_mat), Some(ah)) = (self.x.as_ref(), alpha_hat.as_ref()) {
            let xa = x_mat.dot(ah);
            for i in 0..self.n {
                pred_train[i] += xa[i];
            }
        }
        pred_train.mapv_inplace(|v| v + mu_hat);

        let sigma2_g: f64 = sigma2_g_iter_store.mean().unwrap();
        let h2 = sigma2_g / (sigma2_g + sigma2_e_hat);

        let z_hat = z_samples.as_ref().map(|zs| zs.mean_axis(ndarray::Axis(0)).unwrap());

        if self.verbose {
            eprintln!("[Fold {}] BayesR-SNP done | σ²_g = {:.6} | h² = {:.4}",
                      self.fold_id, sigma2_g, h2);
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn snp_kernel_smoke_gaussian() {
        let n = 50usize;
        let p = 20usize;
        let w: Array2<f64> = Array::from_shape_fn((n, p), |(i, j)| {
            ((i + j) as f64).sin()
        });
        let y: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let wtw_diag: Vec<f64> = (0..p)
            .map(|j| (0..n).map(|i| w[[i, j]].powi(2)).sum::<f64>())
            .collect();
        let mut runner = BayesRSNPRunner::new(
            w, y, wtw_diag, None,
            vec![0.95, 0.02, 0.02, 0.01],
            vec![0.0, 0.0001, 0.001, 0.01],
            1.0, 1.0,
            -1.0, 0.0,
            2.0, 1.0,
            200, 100, 5,
            42, 0, false, false,
        );
        let r = runner.run();
        assert!(r.sigma2_e_hat.is_finite());
        assert!(r.sigma2_g.is_finite());
        assert!(r.h2.is_finite());
        assert!(r.beta_hat.iter().all(|v| v.is_finite()));
        assert!(r.z_hat.is_none());
    }

    #[test]
    fn snp_kernel_smoke_binary() {
        let n = 50usize;
        let p = 20usize;
        let w: Array2<f64> = Array::from_shape_fn((n, p), |(i, j)| {
            ((i + j) as f64).sin()
        });
        // Binary y from sign of sin(i)
        let y: Vec<f64> = (0..n).map(|i| if (i as f64).sin() > 0.0 { 1.0 } else { 0.0 }).collect();
        let wtw_diag: Vec<f64> = (0..p)
            .map(|j| (0..n).map(|i| w[[i, j]].powi(2)).sum::<f64>())
            .collect();
        let mut runner = BayesRSNPRunner::new(
            w, y, wtw_diag, None,
            vec![0.95, 0.02, 0.02, 0.01],
            vec![0.0, 0.0001, 0.001, 0.01],
            1.0, 1.0,                            // sigma2_e_init, sigma2_ah (binary fixes sigma2_e=1)
            -1.0, 0.0,
            2.0, 1.0,
            200, 100, 5,
            42, 0,
            true,                                // is_binary
            false,
        );
        let r = runner.run();
        // Binary forces sigma2_e_hat ≈ 1 (since fixed at 1.0 each iter).
        assert!((r.sigma2_e_hat - 1.0).abs() < 1e-9);
        assert!(r.h2.is_finite() && r.h2 >= 0.0 && r.h2 <= 1.0);
        assert!(r.beta_hat.iter().all(|v| v.is_finite()));
        assert!(r.z_hat.is_some());
        let z = r.z_hat.unwrap();
        assert_eq!(z.len(), n);
        assert!(z.iter().all(|v| v.is_finite()));
    }
}
