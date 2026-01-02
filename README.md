# MasBayes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.78+-orange.svg)](https://www.rust-lang.org/)
[![R](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)

**Rust-based Bayesian genomic prediction package for multi-allelic markers**

MasBayes supports multi-allelic-based markers for genomic prediction, where markers such as haplotypes or microhaplotypes can be used as predictors directly feeding into prediction models without being decomposed into biallelic markers. We implemented the `W_αh` matrix as described by Da, Y. (2015) and developed BayesA and BayesR models specifically for multiallelic markers. Both matrix constructions and Bayesian models were built on Rust programming to optimise computational efficiency rather than purely using the R implementation. In addition, we also implemented marginalized Gibbs sampling for Bayesian models to reduce correlation between parameters within the MCMC chain and hasten convergence, while baseline allele dropping was also implemented to only estimate informative alleles per haplotype block.

---

## Theoretical background

### $W_αh$ matrix construction

The most important thing in genomic prediction is telling the computer which genetic variations in a population are most significant. We need numerical values that clearly represent how different each individual is from the population average. The $W_αh$ matrix does exactly this by converting DNA sequences (genotypes) into standardized numbers that capture both the presence/absence of specific alleles and how rare or common those alleles are in the population.

Imagine you have a population where most individuals carry allele A, but a few carry rare allele B. When we see someone with allele B, this is more "informative" about their genetics than seeing allele A (which everyone has). A specialized coding system by [Da. Y. (2015)](https://link.springer.com/article/10.1186/s12863-015-0301-1) gives larger deviation values to rare alleles and smaller values to common ones, making rare variants more influential in predictions while keeping the math balanced.

Here, we apply a coding rure. For allele *k* with population frequency *p_k*, individual *i* is coded as:

$$
W_{i,k} = \begin{cases}
  -2(1-p_k) & \text{if genotype } k/k \text{ (homozygous: both copies are } k\text{)} \\
  -(1-2p_k) & \text{if genotype } k/\ell \text{ (heterozygous: one copy of } k\text{)} \\
  2p_k & \text{if genotype } \ell/m \text{ (non-carrier: zero copies of } k\text{)}
\end{cases}
$$

where $k \neq \ell \neq m$ are distinct alleles. Note that we drop the most frequent allele (baseline) from each block, keeping only $h-1$ informative alleles.

Allele frequencies are calculated from phased haplotypes (the two DNA copies each individual inherited from their parents), so tools like [Beagle](https://faculty.washington.edu/browning/beagle/beagle.html) (population-based phasing/imputation) and [FImpute](https://animalbiosciences.uoguelph.ca/~msargol/fimpute/) (pedigree-based phasing/imputation) are crucial. For each haplotype block, we count how many times each allele appears across all individuals and divide by total haplotypes (2n for n individuals). For example, if allele 3 appears in 5 out of 200 haplotypes (100 individuals), its frequency is $p_k = 5/200 = 0.025$ (2.5%).

This standardization ensures three critical properties. First, the matrix is mean-centered with $\mathbb{E}[W_k] = 0$, meaning positive and negative deviations balance out across the population. Second, variance scales with $\text{Var}(W_k) \propto 2p_k(1-p_k)$, matching Hardy-Weinberg genetic expectations where intermediate-frequency alleles contribute most variance. Third, the genomic relationship matrix $\mathbf{G} = \mathbf{W}\mathbf{W}^\top / k_{\alpha h}$ (where $k_{\alpha h} = \text{tr}(\mathbf{G}) / n$) becomes comparable to SNP-based GRM by [VanRaden. (2008)](https://www.journalofdairyscience.org/article/S0022-0302(08)70990-1/fulltext), enabling proven statistical methods like GBLUP and Bayesian alphabets to work directly with multi-allelic markers.

#### Example

Consider 4 individuals genotyped at haplotype block 1_1 containing 4 alleles. Allele 2 is most frequent (60%, baseline allele, dropped). The remaining alleles have frequencies: allele 1 at 37.5%, allele 3 at 2.5% (rare), and allele 4 at 0% in this sample.

Step 1: Phased genotypes (maternal/paternal)
- ID1: 1/3 → carries alleles 1 and 3
- ID2: 1/2 → carries allele 1 and baseline
- ID3: 2/2 → only baseline (homozygous)
- ID4: 3/3 → only allele 3 (homozygous)

Step 2: Apply coding rule

|     | Genotype | allele1 calculation | allele1 value | allele3 calculation | allele3 value |
|-----|----------|---------------------|---------------|---------------------|---------------|
| ID1 | 1/3      | -(1-2×0.375) = -0.25 | **-0.25** | -(1-2×0.025) = -0.95 | **-0.95** |
| ID2 | 1/2      | -(1-2×0.375) = -0.25 | **-0.25** | 2×0.025 = 0.05 | **0.05** |
| ID3 | 2/2      | 2×0.375 = 0.75 | **0.75** | 2×0.025 = 0.05 | **0.05** |
| ID4 | 3/3      | 2×0.375 = 0.75 | **0.75** | -2(1-0.025) = -1.95 | **-1.95** |

Step 3: Final $W_αh$ matrix

|     | hap_1_1_allele1 | hap_1_1_allele3 |
|-----|-----------------|-----------------|
| ID1 | -0.25           | -0.95           |
| ID2 | -0.25           | 0.05            |
| ID3 | 0.75            | 0.05            |
| ID4 | 0.75            | -1.95           |

For rare allele 3 (2.5% frequency): Individuals carrying it get large negative values (ID1: -0.95 heterozygous, ID4: -1.95 homozygous), making them stand out strongly from the population. Non-carriers get small positive values (0.05), barely different from average. 

For common allele 1 (37.5% frequency): Carriers get moderate negative values (-0.25 for heterozygous), while non-carriers get moderate positive values (0.75). The deviations are smaller because this allele is common and less "informative."

This weighting ensures rare, potentially high-impact genetic variants contribute more to genomic predictions than common background variation.

However, compared to biallelic markers (SNPs) that have a simple 0/1/2 coding and only one effect per marker to estimate, multi-allelic markers, such as haplotypes and microhaplotypes, have a more expanded parameter space. Each haplotype block can have multiple alleles (often 10-20 or more), and after dropping the baseline, we must estimate separate effects for each remaining allele. So, while the linear model equation seems similar:

$$
\mathbf{y} = \mathbf{W}_{\alpha h}\boldsymbol{\beta}_{\alpha h} + \mathbf{e}
$$

where $\mathbf{y}$ is the phenotype vector, $W_{\alpha h}$ is our multi-allelic matrix, $\boldsymbol{\beta_{\alpha h}}$ contains all allele effects, and $\mathbf{e}$ is residual error. The key difference is that $\boldsymbol{\beta}$ now has thousands or tens of thousands of parameters instead of hundreds of thousands of SNPs, but each parameter carries more biological information.

The challenge with many alleles per block is that we need smarter statistical models that can automatically identify which alleles truly affect the trait versus those that are just noise. Moreover, these models should be able to share information across alleles, since if one allele in a block has a large effect, nearby alleles might too. Most importantly, they must handle sparsity effectively, recognizing that most alleles probably have zero or tiny effects while a few might be critically important.

This is where Bayesian variable selection methods excel. Instead of forcing all allele effects to shrink equally like GBLUP does, or selecting a fixed number like LASSO, Bayesian approaches use mixture models that let the data decide which alleles matter. Think of it this way: in a population, genetic effects follow a pattern where most variants do almost nothing, some have small effects, and a few rare ones have large impacts. Traditional methods struggle because they can't capture this natural diversity.

Bayesian mixture models solve this problem through different strategies. BayesR assigns each allele to one of four "effect size categories": zero, small, medium, or large, with probabilities learned from the data. It's essentially saying "this allele is probably in the large-effect group" or "this one is probably noise," so the model can adapt to the true genetic architecture. In contrast, BayesA gives each allele its own variance parameter, enabling truly flexible shrinkage where alleles with strong evidence get large variances and stay in the model, while weak ones get shrunk to near-zero.

Both methods use the Markov Chain Monte Carlo (MCMC) algorithm to explore the space of possible effect sizes, gradually learning which alleles belong where through iterative sampling. The beauty of these approaches is that they don't just give point estimates but provide full posterior distributions, quantifying uncertainty in every prediction and offering deeper insights into the genetic architecture underlying complex traits.

So here, we extend Bayesian models for multiallelic markers, that may offer superior predictive ability compared to traditional SNP-based approaches while (hopefully) dramatically reducing genotyping costs. By capturing multiple mutations within haplotype blocks as single genetic units, these models can detect functional variants that biallelic SNPs might miss or require dozens of markers to tag. Furthermore, the reduced number of parameters, from hundreds of thousands of SNPs to tens of thousands of alleles, makes these methods computationally efficient while maintaining biological interpretability, as each allele effect represents a natural evolutionary unit rather than an arbitrary single-nucleotide change.

---

## BayesR mixture model

### The core idea: Categorizing allele effects

BayesR recognizes that in real biological systems, genetic variants don't all behave the same way. Some alleles have essentially zero effect on the trait, others have small effects, some have medium effects, and a rare few have large effects. Rather than forcing all alleles to follow the same statistical distribution, BayesR lets each allele belong to one of four categories, each with its own variance.

The model works hierarchically, building from simple to complex:

**Level 1: Phenotype depends on allele effects**

$$
y \mid \boldsymbol{\beta}, \sigma^2_e \sim N(\mathbf{W}\boldsymbol{\beta}, \sigma^2_e \mathbf{I})
$$

Our observed phenotype $y$ is simply the sum of all allele effects $\mathbf{W}\boldsymbol{\beta}$ plus some random environmental noise $\sigma^2_e$.

**Level 2: Each allele effect comes from one of four categories**

$$
\beta_j \mid \gamma_j, \sigma^2_{\gamma_j} \sim N(0, \sigma^2_{\gamma_j})
$$

Each allele effect $\beta_j$ is drawn from a normal distribution, but which normal distribution? That's determined by $\gamma_j$, a categorical label that says "this allele belongs to category 0, 1, 2, or 3."

**Level 3: Categories have different variances**

$$
\begin{align}
\gamma_j &\sim \text{Categorical}(\boldsymbol{\pi}) \\
\boldsymbol{\pi} &= (\pi_0, \pi_{\text{small}}, \pi_{\text{medium}}, \pi_{\text{large}}) \\
\boldsymbol{\sigma}^2_{\gamma} &= (10^{-8}, \sigma^2_{\text{small}}, \sigma^2_{\text{medium}}, \sigma^2_{\text{large}})
\end{align}
$$

The category assignment $\gamma_j$ is random with probabilities $\boldsymbol{\pi}$. Category 0 gets variance $10^{-8}$ (essentially zero), while categories 1, 2, and 3 get increasingly larger variances. The model learns these variances from the data.

**Level 4: Learn the category properties from data**

$$
\begin{align}
\sigma^2_e &\sim \text{InvGamma}(a_e, b_e) \\
\sigma^2_k &\sim \text{InvGamma}(a_k, b_k) \quad \text{for } k \in \{\text{small, medium, large}\} \\
\boldsymbol{\pi} &\sim \text{Dirichlet}(\boldsymbol{\alpha})
\end{align}
$$

Even the category variances and mixing proportions aren't fixed, they have their own prior distributions. This means the model adapts to our specific data, learning both which alleles belong to which categories and what those categories actually mean in terms of effect sizes.

### Why marginalized Gibbs sampling?

Traditional MCMC for mixture models faces a chicken-and-egg problem: to sample the effect size $\beta_j$, we need to know which category $\gamma_j$ it belongs to. But to assign the category $\gamma_j$, we need to know the effect size $\beta_j$. This creates strong correlation between these two parameters, causing the MCMC chain to explore the parameter space extremely slowly, we call this as "poor mixing."

**Standard Gibbs sampling:**
1. Sample $\beta_j$ assuming we know $\gamma_j$ 
2. Sample $\gamma_j$ assuming we know $\beta_j$
3. Repeat, hoping the chain eventually explores all possibilities

The issue is that if $\beta_j$ is currently large, the sampler is reluctant to switch $\gamma_j$ to a small-effect category, and vice versa. The parameters get "stuck" together.

Instead of this back-and-forth, we use a mathematical trick called marginalization. We integrate out $\beta_j$ completely and ask: "What is the probability that allele $j$ belongs to category $k$, considering all possible values $\beta_j$ could have taken?" This gives us:

$$
p(\gamma_j = k \mid \cdot) = \int p(\gamma_j = k, \beta_j \mid \cdot) \, d\beta_j
$$

The beauty is that this integral has a closed-form solution. After completing the square in the joint distribution, we get:

$$
p(\gamma_j = k \mid \cdot) \propto \pi_k \cdot \left(1 + \lambda_j \rho_{jk}\right)^{-1/2} \cdot \exp\left(\frac{r_j^2 \sigma^2_k}{2\sigma^2_e(\sigma^2_e + \lambda_j \sigma^2_k)}\right)
$$

where:

- $\lambda_j = \mathbf{w}_j^\top \mathbf{w}_j$ measures how much information allele $j$ carries (its "signal strength")
- $r_j = \mathbf{w}_j^\top (\mathbf{y} - \mathbf{W}_{-j}\boldsymbol{\beta}_{-j})$ is the residual correlation between the allele and unexplained phenotype
- $\rho_{jk} = \sigma^2_k/\sigma^2_e$ is the signal-to-noise ratio for category $k$

If we break down this formula, we know that
- $\pi_k$: Prior belief about how common this category is
- $\left(1 + \lambda_j \rho_{jk}\right)^{-1/2}$: Penalty term that prevents overfitting (accounts for model complexity)
- $\exp(\cdots)$: Reward term that increases when allele $j$ explains a lot of residual variance

The model essentially compares four hypotheses for each allele: "Does this allele fit better as zero-effect, small-effect, medium-effect, or large-effect?" The category that best balances explanatory power with parsimony wins.

### Computational implementation

However, computing these probabilities directly can cause numerical underflow (numbers too small to represent). We therefore work in log-space:

$$
\log p(\gamma_j = k \mid \cdot) = \log \pi_k - \frac{1}{2}\log(1 + \lambda_j \rho_{jk}) + \frac{r_j^2 \sigma^2_k}{2\sigma^2_e(\sigma^2_e + \lambda_j \sigma^2_k)}
$$

Then use the log-sum-exp trick for normalization:

$$
p(\gamma_j = k) = \frac{\exp(\log p_k - \max_k \log p_k)}{\sum_{k'} \exp(\log p_{k'} - \max_k \log p_k)}
$$

So by subtracting the maximum log-probability before exponentiating, we can prevent overflow/underflow and ensure numerical stability even with extreme values.

**Sampling procedure:**

1. **Sample category** $\gamma_j$ from the marginalized probabilities above
2. **Sample effect** $\beta_j$ conditional on the chosen category:

$$
\beta_j \mid \gamma_j = k, \cdot \sim N\left(\mu_j, v_j\right)
$$

where:

$$
v_j = \frac{\sigma^2_e \sigma^2_k}{\sigma^2_e + \lambda_j \sigma^2_k}, \quad \mu_j = \frac{r_j \sigma^2_k}{\sigma^2_e + \lambda_j \sigma^2_k}
$$

This two-step process breaks the correlation between $\beta_j$ and $\gamma_j$, dramatically improving MCMC mixing and convergence speed. Our Rust implementation exploits this efficiency, processing thousands of alleles per second.

---

## BayesA Model

### Hierarchical Model

$$
\begin{align}
y \mid \boldsymbol{\beta}, \sigma^2_e &\sim N(\mathbf{W}\boldsymbol{\beta}, \sigma^2_e \mathbf{I}) \\
\beta_j \mid \sigma^2_j &\sim N(0, \sigma^2_j) \\
\sigma^2_j &\sim \text{ScaledInvChiSq}(\nu, S^2)
\end{align}
$$

### Hyperprior

$$
\sigma^2_e \sim \text{InvGamma}(a_e, b_e)
$$

### Marginalized Gibbs sampling

BayesA also benefits from marginalized Gibbs sampling, though the marginalization is over the marker-specific variance rather than the component assignment.

**Step 1: Sample marker effects**

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
\lambda_j = \mathbf{w}_j^\top \mathbf{w}_j \text{ and } r_j = \mathbf{w}_j^\top (\mathbf{y} - \mathbf{W}_{-j}\boldsymbol{\beta}_{-j})
$$

**Step 2: Sample marker-specific variances**

The marker-specific variance is updated from its full conditional:

$$
\sigma^2_j \mid \beta_j, \cdot \sim \text{InvGamma}\left(\frac{\nu + 1}{2}, \frac{\nu S^2 + \beta_j^2}{2}\right)
$$

### Interpretation

- The scaled inverse chi-squared prior provides a natural conjugate structure
- Each marker "learns" its own variance from the data
- Markers with large effects get assigned large variances
- Markers with small effects get shrunk toward zero
- The hyperparameter $\nu$ controls the degrees of freedom: smaller values allow more variance heterogeneity

### Incremental Updates

To avoid recomputing $\mathbf{W}\boldsymbol{\beta}$ from scratch at each iteration, we use incremental updates:

$$
\mathbf{W}\boldsymbol{\beta}^{(t)} = \mathbf{W}\boldsymbol{\beta}^{(t-1)} + \mathbf{w}_j(\beta_j^{(t)} - \beta_j^{(t-1)})
$$

This reduces computational complexity from $O(np)$ to $O(n)$ per marker update.

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

Rust is a low-level programming language, standing at the same level as C++, which is the common backend for many currently existing R packages due to its computational efficiency. Rust, however, provides a strict ownership model and memory safety guarantees to eliminate common bugs like memory leaks and segmentation faults without the need for a garbage collector. Rust also has simpler and more readable syntax than C++.

---

#### 2. R dependencies
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

## Quick start

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