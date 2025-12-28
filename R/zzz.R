.onLoad <- function(libname, pkgname) {
  library.dynam("genomicbayes", pkgname, libname)
}

.onAttach <- function(libname, pkgname) {
  version <- utils::packageVersion("genomicbayes")
  
  packageStartupMessage(
    "\n═══════════════════════════════════════════════════════════════\n",
    " genomicbayes v", version, "\n",
    " Rust-Accelerated Bayesian Genomic Prediction\n",
    "───────────────────────────────────────────────────────────────\n",
    " Features:   Multi-allelic markers\n",
    " Methods:    BayesR (mixture) | BayesA (marker-specific)\n",
    "───────────────────────────────────────────────────────────────\n",
    " Functions:    run_bayesr_mcmc() | run_bayesa_mcmc()\n",
    " Quick start:  ?genomicbayes\n",
    " Report bugs:  github.com/bowo1698/genomicbayes/issues\n",
    "═══════════════════════════════════════════════════════════════\n"
  )
}

.onUnload <- function(libpath) {
  library.dynam.unload("genomicbayes", libpath)
}