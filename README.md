<a id="readme-top"></a>
# MasBayes

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Rust](https://img.shields.io/badge/Rust-1.9+-orange.svg)](https://www.rust-lang.org/)
[![R](https://img.shields.io/badge/R-4.4+-blue.svg)](https://www.r-project.org/)
[![Examples](https://img.shields.io/badge/Examples-Click%20Here-blue)](examples/)

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#masbayes">MasBayes</a></li>
    <li>
      <a href="#installation">Installation</a>
      <ul>
        <li><a href="#direct-r-binary">Direct R Binary</a></li>
        <li><a href="#manual-compiling-via-cargo">Manual Compiling via Cargo</a></li>
      </ul>
    </li>
    <li>
      <a href="#theoretical-background">Theoretical Background</a>
      <ul>
        <li><a href="#w_αh-matrix-construction">W_αh Matrix Construction</a></li>
        <li>
          <a href="#bayesr-mixture-model">BayesR Mixture Model</a>
          <ul>
            <li><a href="#the-core-idea-categorizing-allele-effects">The Core Idea</a></li>
            <li><a href="#why-marginalised-gibbs-sampling">Why Marginalised Gibbs Sampling</a></li>
            <li><a href="#computational-implementation">Computational Implementation</a></li>
          </ul>
        </li>
        <li>
          <a href="#bayesa-model">BayesA Model</a>
          <ul>
            <li><a href="#hierarchical-model">Hierarchical Model</a></li>
            <li><a href="#marginalized-gibbs-sampling">Marginalized Gibbs Sampling</a></li>
          </ul>
        </li>
      </ul>
    </li>
    <li>
      <a href="#quick-start">Quick Start</a>
      <ul>
        <li><a href="#basic-continuous-trait-example">Basic Continuous Trait</a></li>
        <li><a href="#binary-trait-with-albert-chib">Binary Trait</a></li>
        <li><a href="#snp-vs-mh-simulation-proof-of-concept">SNP vs MH Simulation</a></li>
      </ul>
    </li>
    <li><a href="#advanced-usage">Advanced Usage</a></li>
    <li><a href="#want-to-help-us">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>
</details>

### Built With

* [![Rust][Rust]][Rust-url]

## Bayesian genomic prediction for multi-allelic markers

MasBayes natively supports multi-allelic (and biallelic SNP) markers for genomic prediction. Genetic markers such as haplotypes or microhaplotypes can be used as predictors directly feeding into prediction models without being decomposed into biallelic markers. We implemented the $W_{\alpha h}$ matrix as described by [Da, Y. (2015)](https://doi.org/10.1186/s12863-015-0301-1) and developed BayesA and BayesR models specifically for multiallelic markers. Both matrix constructions and Bayesian models were built on Rust programming to optimise computational efficiency rather than purely using the R implementation. In addition, we also implemented marginalised Gibbs sampling for Bayesian models to reduce correlation between parameters within the MCMC chain and hasten convergence.

Furthermore, to avoid perfect multicollinearity and ensure a full rank for the design matrix, we treated alleles with the highest frequency at each locus as the baseline reference and excluded them from the matrix construction, but their effects are implicitly captured by the model’s intercept.

> **Why Rust?** Rust is a middle-level programming language, standing at the same level as C++, which is the common backend for many currently existing R packages due to its computational efficiency. Rust, however, has a strict ownership model and memory safety guarantees to eliminate common bugs like memory leaks and segmentation faults without the need for a garbage collector. In addition, we developed this Rust-based library, as Rust implementation is still limited for genomic analysis purposes, while this programming language potentially offers more benefits than others. 

---

## Installation

### Direct R binary

We have provided MasBayes as a ready-to-use package for R, which can be installed in a very convenient way.

```r
# Linux x64 x86
install.packages(
  "https://github.com/bowo1698/masbayes/releases/download/v1.0/masbayes_x64_x86_64-unknown-linux-gnu.tar.gz", 
  repos = NULL
)

# MacOS ARM 64
install.packages(
  "https://github.com/bowo1698/masbayes/releases/download/v1.0/masbayes_arm64_aarch64-apple-darwin.tar.gz", 
  repos = NULL
)

# MacOS x64
install.packages(
  "https://github.com/bowo1698/masbayes/releases/download/v1.0/masbayes_x64_x86_64-apple-darwin.tar.gz", 
  repos = NULL
)

# Windows x64 x86
install.packages(
  "https://github.com/bowo1698/masbayes/releases/download/v1.0/masbayes_x64_x86_64-pc-windows-gnu.zip", 
  repos = NULL
)
```

### Manual compiling via Cargo

But it is **highly recommended** to manually build the Masbayes package, as different architecture may require different Rust library. So, Cargo is the only compiler you can use.

#### 1. Rust Toolchain (Required)
**macOS & Linux**:
```bash
# Install Rust using rustup (one-time)
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

---

#### 2. R dependencies
```r
# Required R packages
install.packages(c("devtools", "Rcpp"))

# Install from GitHub
devtools::install_github("bowo1698/masbayes")
```

---

### Check instalation
```r
# Load and verify
library(masbayes)

# Check available functions
ls("package:masbayes")
?construct_wah_matrix()
#[1] "construct_wah_matrix"  "run_bayesa"  "run_bayesr"
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Theoretical background

### $W_αh$ matrix construction

Biallelic SNP genotype data is generally represented as a simple matrix with values ​​0, 1, or 2 which indicate the number of alleles at a single locus. However, the $W_{ah}$ matrix represents each specific haplotype (multi-allelic) variant found in the population. Consequently, the $W_{ah}$ matrix is ​​able to capture linkage interactions between markers that often better reflect the effects of functional causal variants than single SNP. 

To construct this complex matrix, [Da. Y. (2015)](https://link.springer.com/article/10.1186/s12863-015-0301-1) introduced a special coding system that serves to balance the statistical influence of each variant based on its frequency. This system mathematically assigns larger deviation values ​​to rare alleles and smaller values ​​to common alleles. This aims to ensure that rare variants have a proportional influence on genomic predictions, while ensuring that the average additive effect across the population remains zero to keep the model balanced. Technically, the coding rule is applied to each individual $i$ for haplotype $k$ with population frequency $p_k$ as follows:

$$
W_{i,k} = \begin{cases}
  -2(1-p_k) & \text{if genotype } k/k \text{ (homozygous: both copies are } k\text{)} \\
  -(1-2p_k) & \text{if genotype } k/\ell \text{ (heterozygous: one copy of } k\text{)} \\
  2p_k & \text{if genotype } \ell/m \text{ (non-carrier: zero copies of } k\text{)}
\end{cases}
$$

where $k \neq \ell \neq m$ are distinct alleles. Note that we drop the most frequent allele (baseline) from each block, keeping only $h-1$ alleles to provide unique solutions.

Allele frequencies are calculated from phased haplotypes (the two DNA copies each individual inherited from their parents), so tools like [Beagle](https://faculty.washington.edu/browning/beagle/beagle.html) (population-based phasing/imputation) and [FImpute](https://animalbiosciences.uoguelph.ca/~msargol/fimpute/) (pedigree-based phasing/imputation) are crucial. For each haplotype block, we count how many times each allele appears across all individuals and divide by total haplotypes ($2n$ for $n$ individuals). For example, if allele 3 appears in 5 out of 200 haplotypes (100 individuals), its frequency is $p_k = 5/200 = 0.025$ (2.5%).

This standardisation ensures three critical properties. First, the matrix is mean-centered with $\mathbb{E}[W_k] = 0$, meaning that positive and negative deviations balance out across the population. Second, variance scales with $\text{Var}(W_k) \propto 2p_k(1-p_k)$, matching Hardy-Weinberg genetic expectations where intermediate-frequency alleles contribute most variance. Third, the haplotype genomic relationship matrix $\mathbf{G} = \mathbf{W}\mathbf{W}^\top / k_{\alpha h}$ (where $k_{\alpha h} = \text{tr}(\mathbf{G}) / n$) becomes comparable to SNP-based GRM by [VanRaden. (2008)](https://www.journalofdairyscience.org/article/S0022-0302(08)70990-1/fulltext), enabling proven statistical methods like GBLUP to work directly with multi-allelic markers.

#### Example

Consider 4 individuals genotyped at **Locus 1_1** which has 4 distinct alleles. Allele 2 is the most frequent (60%), chosen as baseline and dropped. Allele 1 (37.5%) and Allele 3 (2.5% - rare) are included in the matrix. Allele 4 (0%) is ignored as it is monomorphic in this sample.

Step 1: Phased genotypes (maternal/paternal)
- ID1: 1/3 → carries two different non-baseline alleles, alleles 1 and 3
- ID2: 1/2 → Carries one non-baseline and one baseline allele, allele 1 and 2 (baseline)
- ID3: 2/2 → Homozygous for the baseline allele (allele 2)
- ID4: 3/3 → Homozygous for the rare allele (allele 3)

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

The matrix $W_{\alpha h}$ uses allele frequencies to "weight" the genotypes. Here is how to interpret the resulting values:

* **Rare alleles (e.g., Allele 3 at 2.5%):** Because this allele is scarce, its presence creates a massive statistical "deviation." 
    - **Carriers** (ID1 & ID4) get large negative values (**-0.95** to **-1.95**), making them stand out sharply. 
    - **Non-carriers** (ID2 & ID3) get a value near zero (**0.05**), meaning they stay close to the population average.

* **Common alleles (e.g., Allele 1 at 37.5%):** Because this allele is common, its presence is less surprising to the model. 
    - **Carriers** (ID1 & ID2) receive moderate negative values (**-0.25**).
    - **Non-carriers** (ID3 & ID4) receive moderate positive values (**0.75**). 
    The gap between carrier and non-carrier is smaller here because the allele is less "informative" than a rare one.

This weighting ensures rare, potentially high-impact genetic variants contribute more to genomic predictions than common background variation.

However, compared to biallelic markers (SNPs) that have a simple 0/1/2 coding and only one effect per marker to estimate, multi-allelic markers, such as haplotypes and microhaplotypes, have a more expanded parameter space. Each haplotype block can have multiple alleles (often 10-20 or more), and after dropping the baseline, we must estimate separate effects for each remaining allele. So, while the linear model equation seems similar:

$$
\mathbf{y} = \mathbf{W}_{\alpha h}\boldsymbol{\beta}_{\alpha h} + \mathbf{e}
$$

where $\mathbf{y}$ is the phenotype vector, $W_{\alpha h}$ is our multi-allelic matrix, $\boldsymbol{\beta_{\alpha h}}$ contains all allele effects, and $\mathbf{e}$ is residual error. The key difference is that $\boldsymbol{\beta}$ now has thousands or tens of thousands of parameters instead of hundreds of thousands of SNPs, but each parameter carries more biological information.

The challenge of having so many alleles per block is that we need a model that can automatically distinguish between influential and insignificant alleles, and also a model that can share information between alleles. Because if one allele has a large effect, other alleles may also have the same effect. This is where Bayesian models excel, as they can distinguish allele effects into several classes, allowing us to more precisely determine which alleles are making a significant contribution.

BayesR, for example, it classifies alleles based on their effect size, ranging from none to very large (four class stratifications), and this model is considered as the most "learning" model. Meanwhile, BayesA only classifies allele effects into large or small effects.

To explore the effects of these alleles, the MCMC sampling algorithm is used. This algorithm works by taking multiple samples of genotype allele data and directly "seeing" how they affect the trait, which is done tens of thousands of times. This allows us to accurately determine the true distribution of allele effects.

Therefore, we extend the use of these Bayesian models to multi-allelic markers. We hope that by grouping genetic markers into small blocks, rather than leaving them alone, we can "exploit" the collective effect of each allele within that block. Furthermore, from a computational perspective, fewer predictors are used, making the process more efficient.

---

## BayesR mixture model

### The core idea: Categorizing allele effects

BayesR recognises that in real biological systems, genetic variants don't all behave the same way. Some alleles have essentially zero effect on the trait, others have small effects, some have medium effects, and a rare few have large effects. Rather than forcing all alleles to follow the same statistical distribution, BayesR lets each allele belong to one of four categories, each with its own variance.

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

### Why marginalised Gibbs sampling?

Traditional MCMC for mixture models faces a chicken-and-egg problem: to sample the effect size $\beta_j$, we need to know which category $\gamma_j$ it belongs to. But to assign the category $\gamma_j$, we need to know the effect size $\beta_j$. This creates strong correlation between these two parameters, causing the MCMC chain to explore the parameter space extremely slowly, we call this as "poor mixing."

**Standard Gibbs sampling:**
1. Sample $\beta_j$ assuming we know $\gamma_j$ 
2. Sample $\gamma_j$ assuming we know $\beta_j$
3. Repeat, hoping the chain eventually explores all possibilities

The issue is that if $\beta_j$ is currently large, the sampler is reluctant to switch $\gamma_j$ to a small-effect category, and vice versa. The parameters get "stuck" together.

Instead of this back-and-forth, we use a mathematical trick called marginalisation, referring to [Gianola et al (2009)](https://academic.oup.com/genetics/article-abstract/183/1/347/6063216?redirectedFrom=fulltext). We integrate out $\beta_j$ completely and ask: "What is the probability that allele $j$ belongs to category $k$, considering all possible values $\beta_j$ could have taken?" This gives us:

$$
p(\gamma_j = k \mid \cdot) = \int p(\gamma_j = k, \beta_j \mid \cdot) \, d\beta_j
$$

The beauty is that this integral has a closed-form solution. After completing the square in the joint distribution, we get:

$$
p(\gamma_j = k \mid \cdot) \propto \pi_k \cdot \left(1 + \lambda_j \rho_{jk}\right)^{-1/2} \cdot \exp\left(\frac{r_j^2 \sigma^2_k}{2\sigma^2_e(\sigma^2_e + \lambda_j \sigma^2_k)}\right)
$$

where,

$$
\lambda_j = \mathbf{w}_j^\top \mathbf{w}_j
$$

measures how much information allele \(j\) carries (its "signal strength"),

$$
\rho_{jk} = \frac{\sigma_k^2}{\sigma_e^2}
$$

is the signal-to-noise ratio for category \(k\), and \(r_j\) is the residual correlation between the allele and unexplained phenotype with:

$$
r_j = \mathbf{w}_j^\top (\mathbf{y} - \mathbf{W}_{-j}\boldsymbol{\beta}_{-j})
$$

If we break down this formula, we know that
- $\pi_k$: Prior belief about how common this category is
- $\left(1 + \lambda_j \rho_{jk}\right)^{-1/2}$: Penalty term that prevents overfitting (accounts for model complexity)
- $\exp(\cdots)$: Reward term that increases when allele $j$ explains a lot of residual variance

The model essentially compares four hypotheses for each allele: "Does this allele fit better as zero-effect, small-effect, medium-effect, or large-effect?" The category that best balances explanatory power with parsimony wins.

### Computational implementation

However, computing these probabilities directly can cause numerical over/under flow (numbers too small or too large to represent). We therefore work in log-space:

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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Quick start
- [Basic continuous trait example](examples/01_basic_continuous.R)
- [Binary trait with Albert-Chib](examples/02_binary_trait.R)  
- [SNP vs MH simulation proof-of-concept](examples/03_snp_vs_mh_simulation.R)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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

## Want to help us?

Contributions are welcome and very beneficial!
You can email me to improve the Rust implementation, add a new model, documentation, benchmarks, or bug reporting. I will appreciate, really!

---

## License

GPL-3 License - see [LICENSE](LICENSE) file

Copyright (c) 2025 Agus Wibowo

---

## Contact

- **Email**: aguswibowo1698@gmail.com

---

## Acknowledgments

### Built With
- [extendr](https://extendr.github.io/) - Rust extensions for R
- [ndarray](https://docs.rs/ndarray/) - N-dimensional arrays in Rust
- [rand](https://docs.rs/rand/) - Random number generation
- [statrs](https://docs.rs/statrs/) - Statistical distributions

### References

- Meuwissen, T. H. E. et al. Prediction of total genetic value using genome-wide dense marker maps. [Genetics 157, 1819–1829 (2001)](https://doi.org/10.1093/genetics/157.4.1819).

- Sorensen, D. and Gianola, D. Likelihood, Bayesian, and MCMC methods in quantitative genetics. [Springer Science & Business Media. (2002)](https://link.springer.com/book/10.1007/b98952)

- Gianola, D. et al. Additive genetic variability and the Bayesian alphabet. [Genetics. 1, 183 (2009)](https://academic.oup.com/genetics/article-abstract/183/1/347/6063216?redirectedFrom=fulltext).

- Erbe, M. et al. Improving accuracy of genomic predictions within and between dairy cattle breeds with imputed high-density single nucleotide polymorphism panels. [J. Dairy Sci. 95, 4114–4129 (2012)](https://doi.org/10.3168/jds.2011-5019).

- Moser, G. et al. Simultaneous discovery, estimation and prediction analysis of complex traits using a Bayesian mixture model. [PLoS Genet. 11, e1004969 (2015)](https://doi.org/10.1371/journal.pgen.1004969).

- Da, Y. Multi-allelic haplotype model based on genetic partition for genomic prediction and variance component estimation using SNP markers. [BMC Genet. 16, 144 (2015)](https://doi.org/10.1186/s12863-015-0301-1).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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
  title = {masbayes: Bayesian model for genomic prediction using multi-allelic markers},
  year = {2025},
  url = {https://github.com/bowo1698/masbayes},
  note = {R package version 4.4.0}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<p align="center">
  <strong>masbayes</strong> - Making genomic prediction faster and saving your money for genotyping🧬
</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Rust]: https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white
[Rust-url]: https://rust-lang.org/