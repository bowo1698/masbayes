# MasBayes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.78+-orange.svg)](https://www.rust-lang.org/)
[![R](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)

**Rust-accelerated Bayesian genomic prediction for multi-allelic markers**

MasBayes supports multi-allelic-based markers for genomic prediction, where markers such as haplotypes or microhaplotypes can be used as predictors directly feeding into prediction models without being decomposed into biallelic markers. We implemented the `W_αh` matrix as described by Da, Y. (2015) and developed BayesA and BayesR models specifically for multiallelic markers. Both matrix constructions and Bayesian models were built on Rust programming to optimise computational efficiency rather than purely using the R implementation. In addition, we also implemented marginalized Gibbs sampling for Bayesian models to reduce correlation between parameters within the MCMC chain and hasten convergence, while baseline allele dropping was also implemented to only estimate informative alleles per haplotype block.

---

## Installation

### Prerequisites

#### 1. Rust Toolchain (Required)
**macOS & Linux**:
```bash
# Install Rust using rustup (one-time, ~5 minutes)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Activate in current session
source $HOME/.cargo/env

# Verify
rustc --version  # Should show: rustc 1.78.0 or higher
cargo --version
```

**Windows**:
1. Download and install [Rustup for Windows](https://rustup.rs/)
2. Install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) (if not already installed)
3. Ensure MSVC toolchain: `rustup default stable-msvc`

**Why Rust?**

Rust is a low-level programming language, standing at the same level as C++, which is the common backend for many currently existing R packages due to its computational efficiency. Rust, however, provides simpler and more readable syntax than C++.

---

#### 2. R Dependencies
```r
# Required R packages
install.packages(c("devtools", "Rcpp"))

# Recommended (for pipeline integration)
install.packages(c("tidyverse", "sommer", "coda"))
```

---

### Install masbayes
```r
# Install from GitHub
devtools::install_github("bowo1698/masbayes")

# Load and verify
library(masbayes)

# Check available functions
ls("package:masbayes")
# [1] "construct_wah_matrix"  "run_bayesa_mcmc"  "run_bayesr_mcmc"
```

---

## Theoretical Background

### W_αh Matrix Coding

For allele *k* with frequency *p_k* in individual *i*:

$$
W_{i,k} = \begin{cases}
  -2(1-p_k) & \text{if genotype } k/k \text{ (homozygous)} \\
  -(1-2p_k) & \text{if genotype } k/\text{other} \text{ (heterozygous)} \\
  2p_k & \text{if genotype other/other}
\end{cases}
$$

This ensures:
- $\mathbb{E}[W_k] = 0$ (centered)
- $\text{Var}(W_k) \propto 2p_k(1-p_k)$ (standardized)

---

### BayesR Mixture Model

**Hierarchical Model:**

$$
\begin{align}
y \mid \boldsymbol{\beta}, \sigma^2_e &\sim N(\mathbf{W}\boldsymbol{\beta}, \sigma^2_e \mathbf{I}) \\
\beta_j \mid \gamma_j, \sigma^2_{\gamma_j} &\sim N(0, \sigma^2_{\gamma_j}) \\
\gamma_j &\sim \text{Categorical}(\boldsymbol{\pi}) \\
\boldsymbol{\pi} &= (\pi_0, \pi_{\text{small}}, \pi_{\text{medium}}, \pi_{\text{large}}) \\
\boldsymbol{\sigma}^2_{\gamma} &= (10^{-8}, \sigma^2_{\text{small}}, \sigma^2_{\text{medium}}, \sigma^2_{\text{large}})
\end{align}
$$

**Hyperpriors:**

$$
\begin{align}
\sigma^2_e &\sim \text{InvGamma}(a_e, b_e) \\
\sigma^2_k &\sim \text{InvGamma}(a_k, b_k) \quad \text{for } k \in \{\text{small, medium, large}\} \\
\boldsymbol{\pi} &\sim \text{Dirichlet}(\boldsymbol{\alpha})
\end{align}
$$

**Marginalized Gibbs Sampling (Rust Implementation):**

Traditional Gibbs sampling alternates between:
1. Sample $\beta_j$ given $\gamma_j$ 
2. Sample $\gamma_j$ given $\beta_j$

This can lead to slow mixing when $\beta_j$ and $\gamma_j$ are highly correlated. Instead, we use **marginalized Gibbs sampling** where we integrate out $\beta_j$ and sample $\gamma_j$ directly from its marginal distribution:

$$
p(\gamma_j = k \mid y, \boldsymbol{\beta}_{-j}, \sigma^2_e, \sigma^2_k) = \int p(\gamma_j = k, \beta_j \mid y, \boldsymbol{\beta}_{-j}, \sigma^2_e, \sigma^2_k) \, d\beta_j
$$

**Step 1: Marginal Distribution for Component Assignment**

By completing the square in the joint distribution, we obtain:

$$
p(\gamma_j = k \mid \cdot) \propto \pi_k \cdot \left(1 + \lambda_j \rho_{jk}\right)^{-1/2} \cdot \exp\left(\frac{r_j^2 \sigma^2_k}{2\sigma^2_e(\sigma^2_e + \lambda_j \sigma^2_k)}\right)
$$

where:

$$
\lambda_j = \mathbf{w}_j^\top \mathbf{w}_j, \quad r_j = \mathbf{w}_j^\top (\mathbf{y} - \mathbf{W}_{-j}\boldsymbol{\beta}_{-j}), \quad \rho_{jk} = \frac{\sigma^2_k}{\sigma^2_e}
$$

**Numerical Stability (Log-Sum-Exp Trick):**

For numerical stability, we compute log-probabilities:

$$
\log p(\gamma_j = k \mid \cdot) = \log \pi_k - \frac{1}{2}\log(1 + \lambda_j \rho_{jk}) + \frac{r_j^2 \sigma^2_k}{2\sigma^2_e(\sigma^2_e + \lambda_j \sigma^2_k)}
$$

Then normalize using the log-sum-exp trick to prevent underflow:

$$
p(\gamma_j = k) = \frac{\exp(\log p_k - \max_k \log p_k)}{\sum_{k'} \exp(\log p_{k'} - \max_k \log p_k)}
$$

**Step 2: Conditional Sampling of Effects**

After sampling $\gamma_j$, we sample $\beta_j$ from its conditional distribution:

$$
\beta_j \mid \gamma_j = k, \cdot \sim N\left(\mu_j, v_j\right)
$$

where:

$$
v_j = \frac{\sigma^2_e \sigma^2_k}{\sigma^2_e + \lambda_j \sigma^2_k}, \quad \mu_j = \frac{r_j \sigma^2_k}{\sigma^2_e + \lambda_j \sigma^2_k}
$$

By using marginalized Gibbs sampling, it can improve mixing as we break the correlation between $\beta_j$ and $\gamma_j$.

---

### BayesA Model

**Hierarchical Model:**

$$
\begin{align}
y \mid \boldsymbol{\beta}, \sigma^2_e &\sim N(\mathbf{W}\boldsymbol{\beta}, \sigma^2_e \mathbf{I}) \\
\beta_j \mid \sigma^2_j &\sim N(0, \sigma^2_j) \\
\sigma^2_j &\sim \text{ScaledInvChiSq}(\nu, S^2)
\end{align}
$$

**Hyperprior:**

$$
\sigma^2_e \sim \text{InvGamma}(a_e, b_e)
$$

**Marginalized Gibbs Sampling:**

BayesA also benefits from marginalized Gibbs sampling, though the marginalization is over the marker-specific variance rather than the component assignment.

**Step 1: Sample Marker Effects**

The conditional posterior for $\beta_j$ is:

$$
\beta_j \mid \sigma^2_j, \cdot \sim N\left(\mu_j, v_j\right)
$$

where:

$$
v_j = \frac{\sigma^2_e \sigma^2_j}{\sigma^2_e + \lambda_j \sigma^2_j}, \quad \mu_j = \frac{r_j \sigma^2_j}{\sigma^2_e + \lambda_j \sigma^2_j}
$$

with 
$$
\lambda_j = \mathbf{w}_j^\top \mathbf{w}_j$ and $r_j = \mathbf{w}_j^\top (\mathbf{y} - \mathbf{W}_{-j}\boldsymbol{\beta}_{-j})
$$.

**Step 2: Sample Marker-Specific Variances**

The marker-specific variance is updated from its full conditional:

$$
\sigma^2_j \mid \beta_j, \cdot \sim \text{InvGamma}\left(\frac{\nu + 1}{2}, \frac{\nu S^2 + \beta_j^2}{2}\right)
$$

**Interpretation:**

- The scaled inverse chi-squared prior provides a natural conjugate structure
- Each marker "learns" its own variance from the data
- Markers with large effects get assigned large variances
- Markers with small effects get shrunk toward zero
- The hyperparameter $\nu$ controls the degrees of freedom: smaller values allow more variance heterogeneity

**Incremental Updates:**

To avoid recomputing $\mathbf{W}\boldsymbol{\beta}$ from scratch at each iteration, we use incremental updates:

$$
\mathbf{W}\boldsymbol{\beta}^{(t)} = \mathbf{W}\boldsymbol{\beta}^{(t-1)} + \mathbf{w}_j(\beta_j^{(t)} - \beta_j^{(t-1)})
$$

This reduces computational complexity from $O(np)$ to $O(n)$ per marker update.

---

## Quick Start

### W_αh matrix construction
```r
library(masbayes)

# Simulated haplotype data (n × 2B matrix)
set.seed(123)
n <- 10  # individuals
B <- 100   # haplotype blocks

hap_matrix <- matrix(
  sample(1:5, n * B * 2, replace = TRUE),
  nrow = n,
  ncol = B * 2
)

# Column names (pairs of haplotypes)
col_names <- paste0("block_", rep(1:B, each = 2))

# Allele frequencies (required for W matrix construction)
# Must be a LIST with: haplotype, allele, freq
allele_freq <- list(
  haplotype = rep(paste0("blk_", 1:B), each = 3),
  allele = rep(1:5, B * 2),
  freq = rep(c(0.6, 0.3, 0.1), B)
)

# Construct W_αh matrix
train_Wah <- construct_wah_matrix(
  hap_matrix = hap_matrix,
  colnames = col_names,
  allele_freq_filtered = allele_freq,
  reference_structure = NULL,
  drop_baseline = TRUE
)

# Output
dim(train_Wah$W_ah)          # n × m (m = total informative alleles)
head(train_Wah$allele_info)  # Metadata for each column
head(train_Wah$dropped_alleles)  # Baseline alleles removed
```

---

### BayesR genomic prediction
```r
# Use the W matrix that has been constructed by construct_wah_matrix()
W <- train_Wah$W_ah
y <- rnorm(n)  # Phenotypes

# Precompute sufficient statistics
wtw_diag <- colSums(W^2)
wty <- as.vector(crossprod(W, y))

# Hyperparameters
prior_params <- list(
  a0_e = 3.0,       # Residual variance prior df
  b0_e = 0.5,      # Residual variance prior scale
  a0_small = 5.0,    # Small effects prior df
  b0_small = 0.001,
  a0_medium = 5.0,   # Medium effects prior df
  b0_medium = 0.005,
  a0_large = 5.0,    # Large effects prior df
  b0_large = 0.1
)

mcmc_params <- list(
  n_iter = as.integer(100), # Total iteration is typically more than 10000
  n_burn = as.integer(20),  # Burn-in can be a half of n_iter
  n_thin = as.integer(2),   # Thinning interval can be 3, 5, or 10
  seed = as.integer(42)
)

# Run BayesR MCMC
result_bayesr <- run_bayesr_mcmc(
  w = W,
  y = y,
  wtw_diag = wtw_diag,
  wty = wty,
  pi_vec = c(0.90, 0.05, 0.03, 0.02),  # Mixture proportions
  sigma2_vec = c(0, 0.001, 0.01, 0.1),  # Variance components
  sigma2_e_init = var(y) * 0.5, # Initial error variance
  sigma2_ah = var(y) * 0.5,  # Initial genetic variance
  prior_params = prior_params,
  mcmc_params = mcmc_params
)

# Posterior means
beta_hat <- colMeans(result_bayesr$beta_samples)
pi_post <- colMeans(result_bayesr$pi_samples)

# Genomic predictions
GEBV <- W %*% beta_hat

# Posterior mixture proportions
pi_post <- colMeans(res$pi_samples)
names(pi_post) <- c("Zero", "Small", "Medium", "Large")
print(pi_post)

# Identify markers with non-zero effects
gamma_mode <- apply(res$gamma_samples, 2, function(x) {
  as.numeric(names(sort(table(x), decreasing = TRUE)[1]))
})
important_markers <- which(gamma_mode > 0)

# Convergence diagnostics
plot(res$sigma2_e_samples, type = "l", main = "Residual Variance Trace")
matplot(res$pi_samples, type = "l", main = "Mixture Proportions", 
        ylab = "Proportion", col = 1:4, lty = 1)
legend("topright", legend = c("Zero", "Small", "Medium", "Large"), 
       col = 1:4, lty = 1)
```

---

### BayesA with marker-specific variance
```r
# Use the W matrix that has been constructed by construct_wah_matrix()
W <- result$W_ah
y <- rnorm(n)  # Phenotypes

# Precompute sufficient statistics
wtw_diag <- as.numeric(colSums(W^2))
wty <- as.vector(crossprod(W, y))

# Prior parameters
prior_params <- list(
  a0_e = 3.0,
  b0_e = var(y) * 0.5 * 2
)

# MCMC parameters
mcmc_params <- list(
  n_iter = as.integer(100), # typically more than 10000
  n_burn = as.integer(20),  # half of n_iter
  n_thin = as.integer(2),   # can be 3, 5, or 10
  seed = as.integer(42)
)

# Run BayesA
result_bayesa <- run_bayesa_mcmc(
  w = W,
  y = y,
  wtw_diag = wtw_diag,
  wty = wty,
  nu = 4.5,                    # Prior df for marker variances
  s_squared = var(y) / ncol(W),  # Prior scale
  sigma2_e_init = var(y) * 0.5,
  prior_params = prior_params,
  mcmc_params = mcmc_params
)

# Posterior inference
beta_hat <- colMeans(result_bayesa$beta_samples)
GEBV <- W %*% beta_hat

# Prediction accuracy
cor(GEBV, y)

# Marker-specific variances
sigma2_j_hat <- colMeans(result_bayesa$sigma2_j_samples)

# Identify markers with large effects
important_markers <- which(sigma2_j_hat > quantile(sigma2_j_hat, 0.95))
```

---

## Advanced usage

### Cross-Validation Pipeline Integration
```r
# Training set
W_result_train <- construct_wah_matrix(
  hap_matrix = hap_train,
  colnames = colnames(hap_train),
  allele_freq_filtered = allele_freq,
  reference_structure = NULL,
  drop_baseline = TRUE
)

# Test set (using reference structure from training)
W_result_test <- construct_wah_matrix(
  hap_matrix = hap_test,
  colnames = colnames(hap_test),
  allele_freq_filtered = NULL,
  reference_structure = W_result_train,  # Same allele structure
  drop_baseline = TRUE
)

# Ensure same number of alleles
ncol(W_result_train$W_ah) == ncol(W_result_test$W_ah)  # TRUE
```

---

## Function reference

### `construct_wah_matrix()`
Fast construction of W_αh design matrix for multi-allelic markers.

**Parameters:**
- `hap_matrix`: Integer matrix (n × 2B) of haplotype genotypes
- `colnames`: Character vector of column names
- `allele_freq_filtered`: List with `haplotype`, `allele`, `freq` (training only)
- `reference_structure`: Reference allele structure (test only)
- `drop_baseline`: Logical, drop most frequent allele per block

**Returns:**
- `W_ah`: Design matrix (n × m)
- `allele_info`: Metadata data frame
- `dropped_alleles`: Baseline alleles data frame

**See:** `?construct_wah_matrix`

---

### `run_bayesr_mcmc()`
BayesR with 4-component mixture prior for variable selection.

**Parameters:**
- `w`: Design matrix (n × p)
- `y`: Phenotype vector (n)
- `wtw_diag`: Diagonal of W'W (p)
- `wty`: W'y vector (p)
- `pi_vec`: Mixture proportions (length 4)
- `sigma2_vec`: Variance components (length 4)
- `sigma2_e_init`: Initial residual variance
- `sigma2_ah`: Initial genetic variance
- `prior_params`: List of prior hyperparameters
- `mcmc_params`: List of MCMC settings

**Returns:**
- `beta_samples`: Posterior samples (n_save × p)
- `gamma_samples`: Component assignments (n_save × p)
- `sigma2_e_samples`: Residual variance samples (n_save)
- `sigma2_*_samples`: Mixture variance samples
- `pi_samples`: Mixture proportion samples (n_save × 4)

**See:** `?run_bayesr_mcmc`

---

### `run_bayesa_mcmc()`
BayesA with marker-specific variance (scaled inverse chi-squared prior).

**Parameters:**
- `w`, `y`, `wtw_diag`, `wty`: Same as BayesR
- `nu`: Prior degrees of freedom for marker variances
- `s_squared`: Prior scale for marker variances
- `sigma2_e_init`: Initial residual variance
- `prior_params`: List with `a0_e`, `b0_e`
- `mcmc_params`: MCMC settings

**Returns:**
- `beta_samples`: Posterior samples (n_save × p)
- `sigma2_j_samples`: Marker variance samples (n_save × p)
- `sigma2_e_samples`: Residual variance samples (n_save)

**See:** `?run_bayesa_mcmc`

---

## Want to help us?

Contributions are welcome and very beneficial!
You can email me to improve the Rust implementation, add a new model, documentation, benchmarks, or bug reporting. I will appreciate, really!

---

## License

MIT License - see [LICENSE](LICENSE) file

Copyright (c) 2025 Agus Wibowo

---

## Support & Contact

- **Email**: aguswibowo1698@gmail.com
- **Documentation**: (coming soon)

---

## Acknowledgments

### Built With
- [extendr](https://extendr.github.io/) - Rust extensions for R
- [ndarray](https://docs.rs/ndarray/) - N-dimensional arrays in Rust
- [rand](https://docs.rs/rand/) - Random number generation
- [statrs](https://docs.rs/statrs/) - Statistical distributions

### References

- Meuwissen, T. H. E. et al. Prediction of total genetic value using genome-wide dense marker maps. [Genetics 157, 1819–1829 (2001)](https://doi.org/10.1093/genetics/157.4.1819).

- Erbe, M. et al. Improving accuracy of genomic predictions within and between dairy cattle breeds with imputed high-density single nucleotide polymorphism panels. [J. Dairy Sci. 95, 4114–4129 (2012)](https://doi.org/10.3168/jds.2011-5019).

- Moser, G. et al. Simultaneous discovery, estimation and prediction analysis of complex traits using a Bayesian mixture model. [PLoS Genet. 11, e1004969 (2015)](https://doi.org/10.1371/journal.pgen.1004969).

- Da, Y. Multi-allelic haplotype model based on genetic partition for genomic prediction and variance component estimation using SNP markers. [BMC Genet. 16, 144 (2015)](https://doi.org/10.1186/s12863-015-0301-1).

---

## Development Team

**Lead Developer:** Agus Wibowo  
James Cook University

**Supervisors:**  
- Prof. Kyall Zenger
- Dr. Cecile Massault

## Citation

If you use `masbayes` in your research, please cite:
```bibtex
@software{masbayes2025,
  author = {Agus Wibowo},
  title = {masbayes: Rust-Accelerated Bayesian Genomic Prediction for Multi-Allelic Markers},
  year = {2025},
  url = {https://github.com/bowo1698/masbayes},
  note = {R package version 1.1.0}
}
```

---

<p align="center">
  <strong>masbayes</strong> - Making genomic prediction faster and saving your money for genotyping🧬
</p>