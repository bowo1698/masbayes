# genomicbayes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Rust-accelerated MCMC backend for BayesR and BayesA genomic prediction methods.

## Features

- Multi-allelic support (only)
- BayesR with mixture prior (4-component)
- BayesA with marker-specific variance

## System Requirements

### Essential
- **R** (>= 4.0.0) - Standard R installation from [CRAN](https://cran.r-project.org/)
- **Rust** (>= 1.70.0) - Required for compilation
- **Cargo** (comes with Rust)

**macOS & Linux** (via Terminal):
```bash
# Install Rust using rustup (recommended method)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the prompts (usually just press Enter)
# Then activate Rust in current session:
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

**Windows**:
1. Download Rust installer from [https://rustup.rs](https://rustup.rs)
2. Run `rustup-init.exe`
3. Follow installation prompts (use default settings)
4. Restart your terminal/command prompt
5. Verify installation:

```cmd
   rustc --version
   cargo --version
```

**Note:** Rust installation is a one-time setup (~5 minutes, ~500MB). Once installed, you can build this and other Rust-based R packages.

---

### Additional Platform Requirements

These are usually already installed, but may be needed:

**macOS**:
```bash
# Xcode Command Line Tools (if not already installed)
xcode-select --install
```

**Linux**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential

# RHEL/CentOS/Fedora
sudo yum groupinstall "Development Tools"
```

**Windows**:
- [Rtools](https://cran.r-project.org/bin/windows/Rtools/) - for R package compilation
- Rust installer will handle the rest

---

## Installation

Once Rust is installed, you can install `genomicbayes`:

### Install from GitHub
```r
# Install devtools if not already installed
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
}

# Install genomicbayes
devtools::install_github("bowo1698/genomicbayes")
```

### Verify Installation
```r
library(genomicbayes)

# Check available functions
ls("package:genomicbayes")
# Should show: run_bayesa_mcmc, run_bayesr_mcmc
```

## Quick Start
```r
library(genomicbayes)

# Prepare data
W <- matrix(rnorm(100 * 50), 100, 50)  # Genotype matrix
y <- rnorm(100)                         # Phenotypes
wtw_diag <- colSums(W^2)
wty <- as.vector(t(W) %*% y)

# Set hyperparameters
prior_params <- list(
  a0_e = 10, b0_e = 1,
  a0_small = 5, b0_small = 0.01,
  a0_medium = 5, b0_medium = 0.1,
  a0_large = 5, b0_large = 1
)

mcmc_params <- list(
  n_iter = 10000,
  n_burn = 5000,
  n_thin = 10,
  seed = 123
)

# Run BayesR
result <- run_bayesr_mcmc(
  w = W,
  y = y,
  wtw_diag = wtw_diag,
  wty = wty,
  pi_vec = c(0.9, 0.05, 0.03, 0.02),
  sigma2_vec = c(0, 0.001, 0.01, 0.1),
  sigma2_e_init = 1.0,
  prior_params = prior_params,
  mcmc_params = mcmc_params
)

# Access results
str(result)
```

## Troubleshooting

### Cargo not found
```bash
# Check if cargo is in PATH
which cargo

# If not found, add to PATH (Linux/macOS)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Or in R session
Sys.setenv(PATH = paste(Sys.getenv("PATH"), 
                        path.expand("~/.cargo/bin"), 
                        sep = ":"))
```

### Build fails on macOS
```bash
# Install Command Line Tools
xcode-select --install

# Update Rust
rustup update
```

### Build fails on Windows

- Ensure [Rtools](https://cran.r-project.org/bin/windows/Rtools/) is installed
- Ensure Rust is installed with MSVC toolchain:
```bash
  rustup default stable-msvc
```

## Citation

If you use this package in your research, please cite:
```bibtex
@software{genomicbayes2025,
  author = {Agus Wibowo},
  title = {genomicbayes: Rust-Accelerated Genomic Prediction},
  year = {2025},
  url = {https://github.com/bowo1698/genomicbayes}
}
```

## License

MIT License - see [LICENSE](LICENSE) file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

- Report bugs: [GitHub Issues](https://github.com/bowo1698/genomicbayes/issues)
- Email: aguswibowo1698@gmail.com
- Documentation: [Wiki](https://github.com/bowo1698/genomicbayes/wiki)

## Acknowledgments

Built with:
- [extendr](https://extendr.github.io/) - R bindings for Rust
- [ndarray](https://docs.rs/ndarray/) - Rust array library
- [rand](https://docs.rs/rand/) - Random number generation
```