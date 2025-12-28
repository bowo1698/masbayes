# Package initialization
.onLoad <- function(libname, pkgname) {
  # Load Rust library
  library.dynam("genomicbayes", pkgname, libname)
  
  # Print startup message
  packageStartupMessage("genomicbayes: Rust-accelerated Bayesian genomic prediction")
}

.onUnload <- function(libpath) {
  library.dynam.unload("genomicbayes", libpath)
}
