#include <stdint.h>
#include <Rinternals.h>
#include <R_ext/Parse.h>

// Declare Rust init function
void R_init_masbayes_extendr_extendr(DllInfo *dll);

// R package init
void R_init_masbayes(DllInfo *dll) {
    R_init_masbayes_extendr_extendr(dll);
}