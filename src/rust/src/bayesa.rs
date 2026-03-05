use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use crate::utils::{rinvgamma, rnorm};
use crate::types::BayesAResults;
use crate::utils;

pub struct BayesARunner {
    // Haplotype
    w_hap: Option<Array2<f64>>,
    wtw_diag_hap: Option<Array1<f64>>,
    wty_hap: Option<Array1<f64>>,
    n_hap_alleles: usize,
    beta_hap: Option<Array1<f64>>,
    sigma2_j_hap: Option<Array1<f64>>,

    // SNP additive
    w_snp: Option<Array2<f64>>,
    wtw_diag_snp: Option<Array1<f64>>,
    wty_snp: Option<Array1<f64>>,
    n_snp: usize,
    beta_snp: Option<Array1<f64>>,
    sigma2_j_snp: Option<Array1<f64>>,

    // Shared
    y: Array1<f64>,
    n: usize,
    nu: f64,
    s_squared: f64,
    a0_e: f64,
    b0_e: f64,
    n_iter: usize,
    n_burn: usize,
    n_thin: usize,
    rng: Pcg64,
    sigma2_e: f64,
    mu: f64,
    fold_id: i32,
}

impl BayesARunner {
    pub fn new(
        w_hap: Option<Array2<f64>>,
        w_snp: Option<Array2<f64>>,
        y: Vec<f64>,
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
    ) -> Self {
        let n = w_hap.as_ref()
            .or(w_snp.as_ref())
            .map(|w| w.nrows())
            .expect("Minimal satu matrix harus ada");

        let rng = Pcg64::seed_from_u64(seed);
        let y_arr = Array1::from_vec(y.clone());

        let compute_wtw_wty = |w: &Array2<f64>| -> (Array1<f64>, Array1<f64>) {
            let m = w.ncols();
            let wtw = (0..m).map(|j| w.column(j).iter().map(|x| x * x).sum()).collect();
            let wty = (0..m).map(|j| {
                w.column(j).iter().zip(y_arr.iter()).map(|(a, b)| a * b).sum()
            }).collect();
            (Array1::from_vec(wtw), Array1::from_vec(wty))
        };

        let (wtw_diag_hap, wty_hap, n_hap_alleles, beta_hap, sigma2_j_hap) =
            if let Some(ref w) = w_hap {
                let m = w.ncols();
                let (wtw, wty) = compute_wtw_wty(w);
                (Some(wtw), Some(wty), m,
                 Some(Array1::<f64>::zeros(m)),
                 Some(Array1::<f64>::from_elem(m, s_squared)))
            } else {
                (None, None, 0, None, None)
            };

        let (wtw_diag_snp, wty_snp, n_snp, beta_snp, sigma2_j_snp) =
            if let Some(ref w) = w_snp {
                let m = w.ncols();
                let (wtw, wty) = compute_wtw_wty(w);
                (Some(wtw), Some(wty), m,
                 Some(Array1::<f64>::zeros(m)),
                 Some(Array1::<f64>::from_elem(m, s_squared)))
            } else {
                (None, None, 0, None, None)
            };
        
        let y_mean = y_arr.mean().unwrap_or(0.0);

        Self {
            w_hap,
            wtw_diag_hap,
            wty_hap,
            n_hap_alleles,
            beta_hap,
            sigma2_j_hap,
            w_snp,
            wtw_diag_snp,
            wty_snp,
            n_snp,
            beta_snp,
            sigma2_j_snp,
            y: y_arr,
            n,
            nu,
            s_squared,
            a0_e,
            b0_e,
            n_iter,
            n_burn,
            n_thin,
            rng,
            sigma2_e: sigma2_e_init,
            mu: y_mean,
            fold_id,
        }
    }

    fn sample_effects(
        w: &Array2<f64>,
        wtw_diag: &Array1<f64>,
        wty: &Array1<f64>,  
        beta: &mut Array1<f64>,
        sigma2_j: &mut Array1<f64>,
        fitted: &mut Array1<f64>,
        sigma2_e: f64,
        nu: f64,
        s_squared: f64,
        rng: &mut Pcg64,
        n: usize,
    ) {
        let inv_sigma2_e = 1.0 / sigma2_e;

        for j in 0..w.ncols() {
            let l_j = wtw_diag[j];

            // RHS = W_j'y - W_j'fitted + l_j * beta_j
            let mut wtf: f64 = 0.0;
            for i in 0..n {
                wtf += w[[i, j]] * fitted[i];
            }
            let rhs = wty[j] - wtf + l_j * beta[j];

            let inv_var_post = l_j * inv_sigma2_e + 1.0 / sigma2_j[j];
            let var_post = 1.0 / inv_var_post;
            let mu_post = rhs * inv_sigma2_e * var_post;

            let beta_old = beta[j];
            beta[j] = rnorm(rng, mu_post, var_post.sqrt());

            let delta = beta[j] - beta_old;
            if delta != 0.0 {
                for i in 0..n {
                    fitted[i] += w[[i, j]] * delta;
                }
            }

            let shape_j = (nu + 1.0) / 2.0;
            let scale_j = (nu * s_squared + beta[j].powi(2)) / 2.0;
            sigma2_j[j] = rinvgamma(rng, shape_j, scale_j);
        }
    }

    pub fn run(&mut self) -> BayesAResults {
        let n_save = (self.n_iter - self.n_burn) / self.n_thin;
        let mut mu_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_e_samples = Array1::<f64>::zeros(n_save);

        let mut beta_hap_samples = self.w_hap.as_ref()
            .map(|_| Array2::<f64>::zeros((n_save, self.n_hap_alleles)));
        let mut sigma2_j_hap_samples = self.w_hap.as_ref()
            .map(|_| Array2::<f64>::zeros((n_save, self.n_hap_alleles)));
        let mut beta_snp_samples = self.w_snp.as_ref()
            .map(|_| Array2::<f64>::zeros((n_save, self.n_snp)));
        let mut sigma2_j_snp_samples = self.w_snp.as_ref()
            .map(|_| Array2::<f64>::zeros((n_save, self.n_snp)));

        let mut save_idx = 0;

        eprintln!("[Fold {}] BayesA MCMC started: {} iterations", self.fold_id, self.n_iter);
        eprintln!("[Fold {}] n_hap_alleles={} | n_snp={}", self.fold_id, self.n_hap_alleles, self.n_snp);
        eprintln!("[Fold {}] ν={:.2} | S²={:.6} | σ²_e={:.6}",
            self.fold_id, self.nu, self.s_squared, self.sigma2_e);

        for iter in 0..self.n_iter {

            // --- Intercept ---
            let mut fitted = Array1::<f64>::from_elem(self.n, self.mu);
            if let (Some(ref w), Some(ref b)) = (&self.w_hap, &self.beta_hap) {
                fitted = fitted + w.dot(b);
            }
            if let (Some(ref w), Some(ref b)) = (&self.w_snp, &self.beta_snp) {
                fitted = fitted + w.dot(b);
            }

            let resid_sum: f64 = self.y.iter().zip(fitted.iter())
                .map(|(yi, fi)| yi - fi)
                .sum();
            let mu_sd = (self.sigma2_e / self.n as f64).sqrt();
            self.mu = rnorm(&mut self.rng, resid_sum / self.n as f64, mu_sd);

            // Rebuild fitted dengan mu baru
            let mut fitted = Array1::<f64>::from_elem(self.n, self.mu);
            if let (Some(ref w), Some(ref b)) = (&self.w_hap, &self.beta_hap) {
                fitted = fitted + w.dot(b);
            }
            if let (Some(ref w), Some(ref b)) = (&self.w_snp, &self.beta_snp) {
                fitted = fitted + w.dot(b);
            }

            // --- Update haplotype effects ---
            if self.w_hap.is_some() {
                let w = self.w_hap.as_ref().unwrap();
                let wtw = self.wtw_diag_hap.as_ref().unwrap();
                let beta = self.beta_hap.as_mut().unwrap();
                let sigma2_j = self.sigma2_j_hap.as_mut().unwrap();
                Self::sample_effects(
                    w, wtw, 
                    self.wty_hap.as_ref().unwrap(),
                    beta, sigma2_j, &mut fitted,
                    self.sigma2_e, self.nu, self.s_squared,
                    &mut self.rng, self.n,
                );
            }

            // --- Update SNP effects ---
            if self.w_snp.is_some() {
                let w = self.w_snp.as_ref().unwrap();
                let wtw = self.wtw_diag_snp.as_ref().unwrap();
                let beta = self.beta_snp.as_mut().unwrap();
                let sigma2_j = self.sigma2_j_snp.as_mut().unwrap();
                Self::sample_effects(
                    w, wtw, 
                    self.wty_snp.as_ref().unwrap(),
                    beta, sigma2_j, &mut fitted,
                    self.sigma2_e, self.nu, self.s_squared,
                    &mut self.rng, self.n,
                );
            }

            // --- Variance sigma2_e ---
            let residuals = &self.y - &fitted;
            let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
            let a_e = self.a0_e + self.n as f64 / 2.0;
            let b_e = self.b0_e + sse / 2.0;
            self.sigma2_e = rinvgamma(&mut self.rng, a_e, b_e);

            // --- Store samples ---
            if iter >= self.n_burn && (iter - self.n_burn) % self.n_thin == 0 {
                mu_samples[save_idx] = self.mu;
                sigma2_e_samples[save_idx] = self.sigma2_e;

                if let (Some(ref mut bs), Some(ref mut ss), Some(ref b), Some(ref s)) =
                    (&mut beta_hap_samples, &mut sigma2_j_hap_samples,
                     &self.beta_hap, &self.sigma2_j_hap) {
                    for j in 0..self.n_hap_alleles {
                        bs[[save_idx, j]] = b[j];
                        ss[[save_idx, j]] = s[j];
                    }
                }
                if let (Some(ref mut bs), Some(ref mut ss), Some(ref b), Some(ref s)) =
                    (&mut beta_snp_samples, &mut sigma2_j_snp_samples,
                     &self.beta_snp, &self.sigma2_j_snp) {
                    for j in 0..self.n_snp {
                        bs[[save_idx, j]] = b[j];
                        ss[[save_idx, j]] = s[j];
                    }
                }
                save_idx += 1;
            }

            // Monitor
            let monitor_interval = (self.n_iter / 10).max(100).min(1000);
            if iter % monitor_interval == 0 {
                let mean_abs_hap = self.beta_hap.as_ref()
                    .map(|b| b.iter().map(|x| x.abs()).sum::<f64>() / b.len() as f64)
                    .unwrap_or(0.0);
                let mean_abs_snp = self.beta_snp.as_ref()
                    .map(|b| b.iter().map(|x| x.abs()).sum::<f64>() / b.len() as f64)
                    .unwrap_or(0.0);
                eprintln!(
                    "[Fold {}] Iter {}/{} | σ²e={:.4} | Mean|β|_hap={:.4} Mean|β|_snp={:.4}",
                    self.fold_id, iter, self.n_iter, self.sigma2_e, mean_abs_hap, mean_abs_snp
                );
            }
        }

        // Diagnostics
        let ess = utils::effective_size(&sigma2_e_samples);
        let geweke = utils::geweke_z(&sigma2_e_samples);
        eprintln!("[Fold {}] ESS: {:.0} | Geweke Z: {:.3}", self.fold_id, ess, geweke);
        eprintln!("[Fold {}] BayesA MCMC completed!", self.fold_id);

        // Posterior means & GEBV
        let mu_hat = mu_samples.mean().unwrap();
        let sigma2_e_hat = sigma2_e_samples.mean().unwrap();

        let mut gebv_train = Array1::<f64>::from_elem(self.n, mu_hat);
        let beta_hat;
        let beta_snp_hat;
        let sigma2_j_hat;
        let sigma2_j_snp_hat;

        if let (Some(ref w), Some(ref bs), Some(ref ss)) =
            (&self.w_hap, &beta_hap_samples, &sigma2_j_hap_samples) {
            let bh = bs.mean_axis(ndarray::Axis(0)).unwrap();
            gebv_train = gebv_train + w.dot(&bh);
            sigma2_j_hat = ss.mean_axis(ndarray::Axis(0)).unwrap();
            beta_hat = bh;
        } else {
            beta_hat = Array1::zeros(0);
            sigma2_j_hat = Array1::zeros(0);
        }

        if let (Some(ref w), Some(ref bs), Some(ref ss)) =
            (&self.w_snp, &beta_snp_samples, &sigma2_j_snp_samples) {
            let bs_hat = bs.mean_axis(ndarray::Axis(0)).unwrap();
            gebv_train = gebv_train + w.dot(&bs_hat);
            beta_snp_hat = Some(bs_hat);
            sigma2_j_snp_hat = Some(ss.mean_axis(ndarray::Axis(0)).unwrap());
        } else {
            beta_snp_hat = None;
            sigma2_j_snp_hat = None;
        }

        let gebv_mean = gebv_train.mean().unwrap();
        let sigma2_g = gebv_train.iter()
            .map(|&g| (g - gebv_mean).powi(2))
            .sum::<f64>() / (self.n as f64 - 1.0);
        let h2 = sigma2_g / (sigma2_g + sigma2_e_hat);

        eprintln!("[Fold {}] σ²_g={:.6} | h²={:.4}", self.fold_id, sigma2_g, h2);

        BayesAResults {
            beta_samples: beta_hap_samples.unwrap_or_else(|| Array2::zeros((n_save, 0))),
            beta_snp_samples,
            sigma2_j_samples: sigma2_j_hap_samples.unwrap_or_else(|| Array2::zeros((n_save, 0))),
            sigma2_j_snp_samples,
            sigma2_e_samples,
            mu_samples,
            beta_hat,
            beta_snp_hat,
            mu_hat,
            sigma2_e_hat,
            sigma2_j_hat,
            sigma2_j_snp_hat,
            gebv_train,
            sigma2_g,
            h2,
        }
    }
}