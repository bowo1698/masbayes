# MasBayes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.78+-orange.svg)](https://www.rust-lang.org/)
[![R](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)

**Rust-accelerated Bayesian genomic prediction for multi-allelic markers**

MasBayes supports multi-allelic-based markers for genomic prediction, where markers such as haplotypes or microhaplotypes can be used as predictors directly feeding into prediction models without being decomposed into biallelic markers. We implemented the `W_αh` matrix as described by Da, Y. (2015) and developed BayesA and BayesR models specifically for multiallelic markers. Both matrix constructions and Bayesian models were built on Rust programming to optimise computational efficiency rather than purely using the R implementation. In addition, we also implemented marginalised Gibbs sampling for Bayesian models to reduce correlation between parameters within the MCMC chain and hasten convergence, while baseline allele dropping was also implemented to only estimate informative alleles per haplotype block.

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

**Why Rust?** Provides 10-100x speedup for genomic computations while maintaining memory safety.

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

## Quick Start

### W_αh matrix construction
```r
library(masbayes)

# Simulated haplotype data (n × 2B matrix)
set.seed(123)
n <- 1000  # individuals
B <- 100   # haplotype blocks

hap_matrix <- matrix(
  sample(1:5, n * B * 2, replace = TRUE),
  nrow = n,
  ncol = B * 2
)

# Column names (pairs of haplotypes)
col_names <- paste0("block_", rep(1:B, each = 2), c("", "_copy"))

# Allele frequencies (pre-filtered for MAF > 0.01)
allele_freq <- list(
  haplotype = rep(col_names, each = 5),
  allele = rep(1:5, B * 2),
  freq = runif(B * 2 * 5, 0.05, 0.95)
)

# Construct W_αh matrix
result <- construct_wah_matrix(
  hap_matrix = hap_matrix,
  colnames = col_names,
  allele_freq_filtered = allele_freq,
  reference_structure = NULL,
  drop_baseline = TRUE
)

# Output
dim(result$W_ah)          # n × m (m = total informative alleles)
head(result$allele_info)  # Metadata for each column
head(result$dropped_alleles)  # Baseline alleles removed
```

---

### BayesR genomic prediction
```r
# Use the W matrix that has been constructed previously
W_train <- result$W_ah
y <- rnorm(n)  # Phenotypes

# Precompute sufficient statistics
wtw_diag <- colSums(W_train^2)
wty <- as.vector(crossprod(W_train, y))

# Hyperparameters
prior_params <- list(
  a0_e = 10,       # Residual variance prior df
  b0_e = 1.0,      # Residual variance prior scale
  a0_small = 5,    # Small effects prior df
  b0_small = 0.001,
  a0_medium = 5,   # Medium effects prior df
  b0_medium = 0.01,
  a0_large = 5,    # Large effects prior df
  b0_large = 0.1
)

mcmc_params <- list(
  n_iter = 50000,  # Total iterations
  n_burn = 20000,  # Burn-in
  n_thin = 10,     # Thinning interval
  seed = 123
)

# Run BayesR MCMC
result_bayes <- run_bayesr_mcmc(
  w = W_train,
  y = y,
  wtw_diag = wtw_diag,
  wty = wty,
  pi_vec = c(0.90, 0.05, 0.03, 0.02),  # Mixture proportions
  sigma2_vec = c(0, 0.001, 0.01, 0.1),  # Variance components
  sigma2_e_init = var(y),
  sigma2_ah = var(y) * 0.5,  # Initial genetic variance
  prior_params = prior_params,
  mcmc_params = mcmc_params
)

# Posterior means
beta_hat <- colMeans(result_bayes$beta_samples)
pi_post <- colMeans(result_bayes$pi_samples)

# Genomic predictions
GEBV <- W_train %*% beta_hat

# Convergence diagnostics
library(coda)
sigma2_e_mcmc <- mcmc(result_bayes$sigma2_e_samples)
effectiveSize(sigma2_e_mcmc)
plot(sigma2_e_mcmc)
```

---

### BayesA with marker-specific variance
```r
# Run BayesA
result_bayesa <- run_bayesa_mcmc(
  w = W_train,
  y = y,
  wtw_diag = wtw_diag,
  wty = wty,
  nu = 4.5,                    # Prior df for marker variances
  s_squared = var(y) / ncol(W_train),  # Prior scale
  sigma2_e_init = var(y) * 0.5,
  prior_params = list(
    a0_e = 10,
    b0_e = var(y) * 0.5 * 9
  ),
  mcmc_params = mcmc_params
)

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

## Theoretical Background

### W_αh Matrix Coding

For allele *k* with frequency *p_k*:
```
W[i,k] = { -2(1-p_k)    if genotype = k/k (homozygous)
         { -(1-2p_k)    if genotype = k/other (heterozygous)
         { 2p_k         if genotype = other/other
```

This ensures:
- E[W_k] = 0 (centered)
- Var[W_k] ∝ 2p_k(1-p_k) (standardized)

---

### BayesR Mixture Model

**Likelihood:**
```
y | β, σ²_e ~ N(Wβ, σ²_e I)
```

**Prior:**
```
β_j | γ_j, σ²_γ ~ N(0, σ²_γ[γ_j])
γ_j ~ Categorical(π)

π = (π_0, π_small, π_medium, π_large)
σ²_γ = (0, σ²_small, σ²_medium, σ²_large)
```

**Hyperpriors:**
```
σ²_e ~ InvGamma(a_e, b_e)
σ²_k ~ InvGamma(a_k, b_k)  for k ∈ {small, medium, large}
π ~ Dirichlet(α)
```

---

### BayesA Model

**Likelihood:**
```
y | β, σ²_e ~ N(Wβ, σ²_e I)
```

**Prior:**
```
β_j | σ²_j ~ N(0, σ²_j)
σ²_j ~ ScaledInvChiSq(ν, S²)
```

**Hyperprior:**
```
σ²_e ~ InvGamma(a_e, b_e)
```

---

## Contributing

Contributions are welcome!
You can email me to improve Rust, add new model implementation, documentation, benchmarks, or bug reporting. I will appreciate!

---

## License

MIT License - see [LICENSE](LICENSE) file

Copyright (c) 2025 Agus Bowo Wibowo

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