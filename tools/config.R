# Validate Rust toolchain and report versions before attempting compilation
source("tools/msrv.R")

env_debug <- Sys.getenv("DEBUG")
env_not_cran <- Sys.getenv("NOT_CRAN")

vendor_exists <- file.exists("src/rust/vendor.tar.xz")

is_not_cran <- env_not_cran != ""
is_debug <- env_debug != ""

if (is_debug) {
  # DEBUG implies a development build; CRAN always requires release
  is_not_cran <- TRUE
  message("Creating DEBUG build.")
}

if (!is_not_cran) {
  message("Building for CRAN.")
}

# -j 2 and --offline are required for CRAN: servers are resource-constrained
# and must not make network calls during compilation
.cran_flags <- ifelse(
  !is_not_cran && vendor_exists,
  "-j 2 --offline",
  ""
)

.profile <- ifelse(is_debug, "", "--release")
.clean_targets <- ifelse(is_debug, "", "$(TARGET_DIR)")
.libdir <- ifelse(is_debug, "debug", "release")

is_windows <- .Platform[["OS.type"]] == "windows"

mv_fp <- ifelse(is_windows, "src/Makevars.win.in", "src/Makevars.in")
mv_ofp <- ifelse(is_windows, "src/Makevars.win", "src/Makevars")

if (file.exists(mv_ofp)) {
  message("Cleaning previous `", mv_ofp, "`.")
  invisible(file.remove(mv_ofp))
}

mv_txt <- readLines(mv_fp)

new_txt <- gsub("@CRAN_FLAGS@", .cran_flags, mv_txt) |>
  gsub("@PROFILE@", .profile, x = _) |>
  gsub("@CLEAN_TARGET@", .clean_targets, x = _) |>
  gsub("@LIBDIR@", .libdir, x = _)

message("Writing `", mv_ofp, "`.")
# Binary mode ensures consistent line endings across platforms (LF on all)
con <- file(mv_ofp, open = "wb")
writeLines(new_txt, con, sep = "\n")
close(con)

message("`tools/config.R` has finished.")
