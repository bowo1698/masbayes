.onLoad <- function(libname, pkgname) {
  library.dynam("masbayes", pkgname, libname)
}

.onAttach <- function(libname, pkgname) {
  version <- utils::packageVersion("masbayes")
  
  packageStartupMessage(
    "\n═══════════════════════════════════════════════════════════════\n",
    " masbayes v", version, "\n",
    " Rust-Accelerated Bayesian Genomic Prediction\n",
    "───────────────────────────────────────────────────────────────\n",
    " Features:   Multi-allelic markers\n",
    " Methods:    BayesR (mixture) | BayesA (marker-specific)\n",
    "───────────────────────────────────────────────────────────────\n",
    " Functions:    run_bayesr_mcmc() | run_bayesa_mcmc()\n",
    " Quick start:  ?masbayes\n",
    " Report bugs:  github.com/bowo1698/masbayes/issues\n",
    "═══════════════════════════════════════════════════════════════\n"
  )
}

.onUnload <- function(libpath) {
  library.dynam.unload("masbayes", libpath)
}