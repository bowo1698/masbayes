use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::Rng;
use rand_pcg::Pcg64;
use crate::utils::{rinvgamma, rdirichlet, rnorm};
use crate::types::BayesRResults;
use crate::utils;

pub struct BayesRRunner {
    // Haplotype
    w_hap: Option<Array2<f64>>,
    wtw_diag_hap: Option<Array1<f64>>,
    wty_hap: Option<Array1<f64>>,
    n_hap_alleles: usize,
    beta_hap: Option<Array1<f64>>,
    gamma_hap: Option<Array1<usize>>,

    // SNP additive
    w_snp: Option<Array2<f64>>,
    wtw_diag_snp: Option<Array1<f64>>,
    wty_snp: Option<Array1<f64>>,
    n_snp: usize,
    beta_snp: Option<Array1<f64>>,
    gamma_snp: Option<Array1<usize>>,

    // Shared
    y: Array1<f64>,
    n: usize,
    pi_vec: Array1<f64>,
    sigma2_vec: Array1<f64>,
    mu: f64,
    a0_e: f64,
    b0_e: f64,
    a0_g: f64,
    b0_g: f64,
    variance_class: Array1<f64>,
    n_iter: usize,
    n_burn: usize,
    n_thin: usize,
    rng: Pcg64,
    sigma2_e: f64,
    fold_id: i32,
}

impl BayesRRunner {
    pub fn new(
        w_hap: Option<Array2<f64>>,
        w_snp: Option<Array2<f64>>,
        y: Vec<f64>,
        pi_vec: Vec<f64>,
        variance_class: Vec<f64>,
        sigma2_e_init: f64,
        sigma2_ah: f64,
        a0_e: f64, b0_e: f64,
        a0_g: f64, b0_g: f64,
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
        let mut init_rng = Pcg64::seed_from_u64(seed);
        let y_arr = Array1::from_vec(y.clone());

        let compute_wtw_wty = |w: &Array2<f64>| -> (Array1<f64>, Array1<f64>) {
            let m = w.ncols();
            let wtw = (0..m).map(|j| w.column(j).iter().map(|x| x * x).sum()).collect();
            let wty = (0..m).map(|j| {
                w.column(j).iter().zip(y_arr.iter()).map(|(a, b)| a * b).sum()
            }).collect();
            (Array1::from_vec(wtw), Array1::from_vec(wty))
        };

        let (wtw_diag_hap, wty_hap, n_hap_alleles, beta_hap, gamma_hap) =
            if let Some(ref w) = w_hap {
                let m = w.ncols();
                let (wtw, wty) = compute_wtw_wty(w);
                let beta = Array1::<f64>::zeros(m);
                (Some(wtw), Some(wty), m, Some(beta), Some(Array1::<usize>::zeros(m)))
            } else {
                (None, None, 0, None, None)
            };

        let (wtw_diag_snp, wty_snp, n_snp, beta_snp, gamma_snp) =
            if let Some(ref w) = w_snp {
                let m = w.ncols();
                let (wtw, wty) = compute_wtw_wty(w);
                let beta = Array1::<f64>::zeros(m);
                (Some(wtw), Some(wty), m, Some(beta), Some(Array1::<usize>::zeros(m)))
            } else {
                (None, None, 0, None, None)
            };

        let n_total = (n_hap_alleles + n_snp).max(1);
        let varg_init = sigma2_ah / n_total as f64;
        let sigma2_vec: Vec<f64> = variance_class.iter().map(|&f| f * varg_init).collect();

        Self {
            w_hap,
            wtw_diag_hap,
            wty_hap,
            n_hap_alleles,
            beta_hap,
            gamma_hap,
            w_snp,
            wtw_diag_snp,
            wty_snp,
            n_snp,
            beta_snp,
            gamma_snp,
            y: y_arr,
            n,
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
            sigma2_e: sigma2_e_init,
            fold_id,
        }
    }

    fn sample_effects(
        w: &Array2<f64>,
        wtw_diag: &Array1<f64>,
        wty: &Array1<f64>,
        beta: &mut Array1<f64>,
        gamma: &mut Array1<usize>,
        fitted: &mut Array1<f64>,
        sigma2_e: f64,
        sigma2_vec: &Array1<f64>,
        pi_vec: &Array1<f64>,
        rng: &mut Pcg64,
        n: usize,
    ) {
        let inv_sigma2_e = 1.0 / sigma2_e;

        for j in 0..w.ncols() {
            let beta_old = beta[j];
            let l_j = wtw_diag[j];

            // RHS = W_j'y - W_j'fitted + l_j * beta_old
            // W_j'fitted dihitung per iterasi, W_j'y sudah precomputed
            let mut wtf: f64 = 0.0;
            for i in 0..n {
                wtf += w[[i, j]] * fitted[i];
            }
            let rhs = wty[j] - wtf + l_j * beta_old;

            // Log-probabilities per komponen
            let mut log_probs = [0.0f64; 4];
            log_probs[0] = pi_vec[0].ln();

            for k in 1..4 {
                let sigma2_k = sigma2_vec[k];
                if sigma2_k < 1e-10 {
                    log_probs[k] = f64::NEG_INFINITY;
                    continue;
                }
                let ratio_var = sigma2_k * inv_sigma2_e;
                let log_det = (1.0 + l_j * ratio_var).ln();
                let quad_term = (rhs.powi(2) * sigma2_k) /
                    (sigma2_e * (sigma2_e + l_j * sigma2_k));
                log_probs[k] = pi_vec[k].ln() - 0.5 * log_det + 0.5 * quad_term;
            }

            // Log-sum-exp normalization
            let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut probs = [0.0f64; 4];
            let mut sum_probs = 0.0;
            for k in 0..4 {
                probs[k] = (log_probs[k] - max_log).exp();
                sum_probs += probs[k];
            }
            for k in 0..4 {
                probs[k] /= sum_probs;
            }

            // Sample komponen
            let u: f64 = rng.gen();
            let mut cumsum = 0.0;
            let mut new_gamma_idx = 0;
            for k in 0..4 {
                cumsum += probs[k];
                if u < cumsum {
                    new_gamma_idx = k;
                    break;
                }
            }
            gamma[j] = new_gamma_idx;

            // Sample beta
            let sigma2_k_chosen = sigma2_vec[new_gamma_idx];
            if sigma2_k_chosen < 1e-10 {
                beta[j] = 0.0;
            } else {
                let inv_var_post = l_j * inv_sigma2_e + 1.0 / sigma2_k_chosen;
                let var_post = 1.0 / inv_var_post;
                let mu_post = rhs * inv_sigma2_e * var_post;
                beta[j] = rnorm(rng, mu_post, var_post.sqrt());
            }

            // Update fitted incremental
            let delta = beta[j] - beta_old;
            if delta != 0.0 {
                for i in 0..n {
                    fitted[i] += w[[i, j]] * delta;
                }
            }
        }
    }

    pub fn run(&mut self) -> BayesRResults {
        let n_save = (self.n_iter - self.n_burn) / self.n_thin;
        let mut mu_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_e_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_small_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_medium_samples = Array1::<f64>::zeros(n_save);
        let mut sigma2_large_samples = Array1::<f64>::zeros(n_save);
        let mut pi_samples = Array2::<f64>::zeros((n_save, 4));

        let mut beta_hap_samples = self.w_hap.as_ref()
            .map(|_| Array2::<f64>::zeros((n_save, self.n_hap_alleles)));
        let mut gamma_hap_samples = self.w_hap.as_ref()
            .map(|_| Array2::<f64>::zeros((n_save, self.n_hap_alleles)));
        let mut beta_snp_samples = self.w_snp.as_ref()
            .map(|_| Array2::<f64>::zeros((n_save, self.n_snp)));
        let mut gamma_snp_samples = self.w_snp.as_ref()
            .map(|_| Array2::<f64>::zeros((n_save, self.n_snp)));

        let mut save_idx = 0;

        eprintln!("[Fold {}] BayesR MCMC started: {} iterations", self.fold_id, self.n_iter);
        eprintln!("[Fold {}] n_hap_alleles={} | n_snp={}", self.fold_id, self.n_hap_alleles, self.n_snp);
        eprintln!("[Fold {}] π=[{:.3},{:.3},{:.3},{:.3}] | σ²_e={:.6}",
            self.fold_id,
            self.pi_vec[0], self.pi_vec[1], self.pi_vec[2], self.pi_vec[3],
            self.sigma2_e);
        
        eprintln!("[Fold {}] sigma2_vec=[{:.2e},{:.2e},{:.2e},{:.2e}]",
            self.fold_id,
            self.sigma2_vec[0], self.sigma2_vec[1], self.sigma2_vec[2], self.sigma2_vec[3]);
        eprintln!("[Fold {}] b0_g={:.6} | a0_g={:.4}",
            self.fold_id, self.b0_g, self.a0_g);
        let sum_abs_beta_hap = self.beta_hap.as_ref()
            .map(|b| b.iter().map(|x| x.abs()).sum::<f64>()).unwrap_or(0.0);
        let sum_abs_beta_snp = self.beta_snp.as_ref()
            .map(|b| b.iter().map(|x| x.abs()).sum::<f64>()).unwrap_or(0.0);
        eprintln!("[Fold {}] sum|beta_hap_init|={:.4} | sum|beta_snp_init|={:.4}",
            self.fold_id, sum_abs_beta_hap, sum_abs_beta_snp);
        let y_mean = self.y.mean().unwrap_or(0.0);
        let y_var = self.y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (self.n as f64 - 1.0);
        let y_sd = y_var.sqrt();
        eprintln!("[Fold {}] y_mean={:.4} | y_sd={:.4} | y_var={:.4}", self.fold_id, y_mean, y_sd, y_var);

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
                let gamma = self.gamma_hap.as_mut().unwrap();
                Self::sample_effects(
                    w, wtw, 
                    self.wty_hap.as_ref().unwrap(),
                    beta, gamma, &mut fitted,
                    self.sigma2_e, &self.sigma2_vec,
                    &self.pi_vec, &mut self.rng, self.n,
                );
            }

            // --- Update SNP effects ---
            if self.w_snp.is_some() {
                let w = self.w_snp.as_ref().unwrap();
                let wtw = self.wtw_diag_snp.as_ref().unwrap();
                let beta = self.beta_snp.as_mut().unwrap();
                let gamma = self.gamma_snp.as_mut().unwrap();
                Self::sample_effects(
                    w, wtw, 
                    self.wty_snp.as_ref().unwrap(),
                    beta, gamma, &mut fitted,
                    self.sigma2_e, &self.sigma2_vec,
                    &self.pi_vec, &mut self.rng, self.n,
                );
            }

            // --- Variance components ---
            let residuals = &self.y - &fitted;
            let sse = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
            let a_e = self.a0_e + self.n as f64 / 2.0;
            let b_e = self.b0_e + sse / 2.0;
            self.sigma2_e = rinvgamma(&mut self.rng, a_e, b_e);

            // Pooled varg dari hap + snp
            let mut varg_sum = 0.0;
            let mut n_nz: usize = 0;
            let mut n_counts = [0usize; 4];

            let accumulate = |beta: &Array1<f64>, gamma: &Array1<usize>,
                              varg_sum: &mut f64, n_nz: &mut usize,
                              n_counts: &mut [usize; 4],
                              variance_class: &Array1<f64>| {
                for j in 0..beta.len() {
                    let comp = gamma[j];
                    n_counts[comp] += 1;
                    if comp > 0 {
                        *varg_sum += beta[j].powi(2) / variance_class[comp];
                        *n_nz += 1;
                    }
                }
            };

            if let (Some(b), Some(g)) = (&self.beta_hap, &self.gamma_hap) {
                accumulate(b, g, &mut varg_sum, &mut n_nz, &mut n_counts, &self.variance_class);
            }
            if let (Some(b), Some(g)) = (&self.beta_snp, &self.gamma_snp) {
                accumulate(b, g, &mut varg_sum, &mut n_nz, &mut n_counts, &self.variance_class);
            }

            let a_g = self.a0_g + n_nz as f64 / 2.0;
            let b_g = self.b0_g + varg_sum / 2.0;
            let varg = rinvgamma(&mut self.rng, a_g, b_g);
            for k in 1..4 {
                self.sigma2_vec[k] = varg * self.variance_class[k];
            }

            // Mixture proportions
            let mut alpha_post = Array1::<f64>::ones(4);
            for k in 0..4 {
                alpha_post[k] += n_counts[k] as f64;
            }
            self.pi_vec = rdirichlet(&mut self.rng, &alpha_post);

            // --- Store samples ---
            if iter >= self.n_burn && (iter - self.n_burn) % self.n_thin == 0 {
                mu_samples[save_idx] = self.mu;
                sigma2_e_samples[save_idx] = self.sigma2_e;
                sigma2_small_samples[save_idx] = self.sigma2_vec[1];
                sigma2_medium_samples[save_idx] = self.sigma2_vec[2];
                sigma2_large_samples[save_idx] = self.sigma2_vec[3];
                for k in 0..4 { pi_samples[[save_idx, k]] = self.pi_vec[k]; }

                if let (Some(ref mut bs), Some(ref mut gs), Some(ref b), Some(ref g)) =
                    (&mut beta_hap_samples, &mut gamma_hap_samples,
                     &self.beta_hap, &self.gamma_hap) {
                    for j in 0..self.n_hap_alleles {
                        bs[[save_idx, j]] = b[j];
                        gs[[save_idx, j]] = g[j] as f64;
                    }
                }
                if let (Some(ref mut bs), Some(ref mut gs), Some(ref b), Some(ref g)) =
                    (&mut beta_snp_samples, &mut gamma_snp_samples,
                     &self.beta_snp, &self.gamma_snp) {
                    for j in 0..self.n_snp {
                        bs[[save_idx, j]] = b[j];
                        gs[[save_idx, j]] = g[j] as f64;
                    }
                }
                save_idx += 1;
            }

            // Monitor
            let monitor_interval = (self.n_iter / 10).max(100).min(1000);
            if iter % monitor_interval == 0 {
                let n_nz_hap = self.gamma_hap.as_ref()
                    .map(|g| g.iter().filter(|&&x| x != 0).count()).unwrap_or(0);
                let n_nz_snp = self.gamma_snp.as_ref()
                    .map(|g| g.iter().filter(|&&x| x != 0).count()).unwrap_or(0);
                eprintln!(
                    "[Fold {}] Iter {}/{} | σ²e={:.4} | π=({:.2},{:.2},{:.2},{:.2}) | NZ_hap={} NZ_snp={}",
                    self.fold_id, iter, self.n_iter, self.sigma2_e,
                    self.pi_vec[0], self.pi_vec[1], self.pi_vec[2], self.pi_vec[3],
                    n_nz_hap, n_nz_snp
                );
            }
        }

        // Diagnostics
        let ess = utils::effective_size(&sigma2_e_samples);
        let geweke = utils::geweke_z(&sigma2_e_samples);
        eprintln!("[Fold {}] ESS: {:.0} | Geweke Z: {:.3}", self.fold_id, ess, geweke);
        eprintln!("[Fold {}] BayesR MCMC completed!", self.fold_id);

        // Posterior means & GEBV
        let mu_hat = mu_samples.mean().unwrap();
        let sigma2_e_hat = sigma2_e_samples.mean().unwrap();

        let mut gebv_train = Array1::<f64>::from_elem(self.n, mu_hat);
        let beta_hat;
        let beta_snp_hat;

        if let (Some(ref w), Some(ref bs)) = (&self.w_hap, &beta_hap_samples) {
            let bh = bs.mean_axis(ndarray::Axis(0)).unwrap();
            gebv_train = gebv_train + w.dot(&bh);
            beta_hat = bh;
        } else {
            beta_hat = Array1::zeros(0);
        }

        if let (Some(ref w), Some(ref bs)) = (&self.w_snp, &beta_snp_samples) {
            let bs_hat = bs.mean_axis(ndarray::Axis(0)).unwrap();
            gebv_train = gebv_train + w.dot(&bs_hat);
            beta_snp_hat = Some(bs_hat);
        } else {
            beta_snp_hat = None;
        }

        let gebv_mean = gebv_train.mean().unwrap();
        let sigma2_g = gebv_train.iter()
            .map(|&g| (g - gebv_mean).powi(2))
            .sum::<f64>() / (self.n as f64 - 1.0);
        let h2 = sigma2_g / (sigma2_g + sigma2_e_hat);

        eprintln!("[Fold {}] σ²_g={:.6} | h²={:.4}", self.fold_id, sigma2_g, h2);

        BayesRResults {
            beta_samples: beta_hap_samples.unwrap_or_else(|| Array2::zeros((n_save, 0))),
            gamma_samples: gamma_hap_samples.unwrap_or_else(|| Array2::zeros((n_save, 0))),
            beta_snp_samples,
            gamma_snp_samples,
            sigma2_e_samples,
            sigma2_small_samples,
            sigma2_medium_samples,
            sigma2_large_samples,
            pi_samples,
            mu_samples,
            beta_hat,
            beta_snp_hat,
            mu_hat,
            sigma2_e_hat,
            gebv_train,
            sigma2_g,
            h2,
        }
    }
}