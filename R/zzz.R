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
    " Features:   Multi-allelic genomic markers (haplotypes or microhaplotypes)\n",
    " Methods:    BayesR (mixture) | BayesA (marker-specific)\n",
    " Matrix:     Matrix W_αh construction specified for multi-allelic markers\n",
    "───────────────────────────────────────────────────────────────\n",
    " Core Functions:\n",
    "   • construct_wah_matrix()  - Design matrix construction\n",
    "   • run_bayesr_mcmc()       - Mixture prior MCMC\n",
    "   • run_bayesa_mcmc()       - Marker-specific variance MCMC\n",
    "───────────────────────────────────────────────────────────────\n",
    " Documentation:  ?masbayes\n",
    " Report bugs:    github.com/bowo1698/masbayes/issues\n",
    "═══════════════════════════════════════════════════════════════\n"
  )
}

.onUnload <- function(libpath) {
  library.dynam.unload("masbayes", libpath)
}