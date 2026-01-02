.onLoad <- function(libname, pkgname) {
  library.dynam("masbayes", pkgname, libname)
}

.onAttach <- function(libname, pkgname) {
  version <- utils::packageVersion("masbayes")
  
  packageStartupMessage(
    "\n═══════════════════════════════════════════════════════════════\n",
    " masbayes v", version, "\n",
    " Bayesian Genomic Prediction models for multi-allelic marker\n",
    "───────────────────────────────────────────────────────────────\n",
    " Methods:    BayesR (mixture) | BayesA (marker-specific)\n",
    " Algorithms: MCMC (full Bayesian) | Stochastic EM (fast)\n",
    "───────────────────────────────────────────────────────────────\n",
    " Core Functions:\n",
    "   • construct_wah_matrix()  - Design matrix construction\n",
    "   • run_bayesr()            - Mixture-based variable selection\n",
    "   • run_bayesa()            - Marker-specific variance modeling\n",
    "───────────────────────────────────────────────────────────────\n",
    " Documentation:  ?masbayes\n",
    " Report bugs:    github.com/bowo1698/masbayes/issues\n",
    "═══════════════════════════════════════════════════════════════\n"
  )
}

.onUnload <- function(libpath) {
  library.dynam.unload("masbayes", libpath)
}