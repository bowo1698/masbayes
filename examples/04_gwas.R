# 04_gwas.R — One-shot genomic prediction + GWAS with BayesR.
#
# Demonstrates the GWAS surface added in masbayes v0.5.0: supplying a `map`
# together with `windsize` (or `windnum`) to run_bayesr() attaches
# per-allele PIP, per-block PIP, and per-window WPPA to the fit object.
# The same call still returns the usual genomic prediction (h2, GEBV,
# beta_hat, posterior chains).
#
# Run with:
#   Rscript examples/04_gwas.R
#
# Uses the bundled load_data() demo (n = 200, 100 SNPs across 50 MH blocks,
# 10 QTL per architecture). The demo ships d$map_snp and d$map_mh (schema
# matches the maspipeline microhaplotype_coordinates.csv output), so the
# example uses them directly — no synthetic map construction needed.

suppressPackageStartupMessages(library(masbayes))

d <- load_data()
mcmc <- list(n_iter = 2000L, n_burn = 1000L, n_thin = 5L, seed = 42L)

# ----- (A) SNP path ---------------------------------------------------------
cat("\n== (A) SNP GWAS ==\n")
y_snp <- d$pheno$y_cont_qtl_snp
snp   <- construct_snp_matrix(d$snp, encoding = "zscore")
W_snp <- snp$W

# Per-SNP map: shipped with the demo (5 chr, 100 kb intra-chr spacing).
map_snp <- d$map_snp

fit_snp <- run_bayesr(
  w             = W_snp,
  y             = y_snp,
  wtw_diag      = colSums(W_snp^2),
  marker_type   = "snp",
  sigma2_e_init = var(y_snp) * 0.5,
  sigma2_ah     = var(y_snp) * 0.5,
  mcmc_params   = mcmc,
  save_rds      = FALSE,
  verbose       = FALSE,
  map           = map_snp,
  windsize      = 5e5
)

cat("Top 10 SNP windows by WPPA:\n")
top10_snp <- fit_snp$gwas[order(-fit_snp$gwas$WPPA), ][seq_len(min(10L, nrow(fit_snp$gwas))), ]
print(top10_snp, row.names = FALSE)

# ----- (B) Microhaplotype path ---------------------------------------------
cat("\n== (B) MH GWAS ==\n")
y_mh  <- d$pheno$y_cont_qtl_mh
bid   <- attr(d$mh, "block_id")
wah   <- construct_wah_matrix(d$mh, bid, d$allele_freq)
W_mh  <- wah$W_ah

# Per-block map: shipped with the demo. Schema matches the maspipeline
# microhaplotype_coordinates.csv (block_id, chr, start_pos, end_pos, n_snps).
map_mh <- d$map_mh

fit_mh <- run_bayesr(
  w             = W_mh,
  y             = y_mh,
  wtw_diag      = colSums(W_mh^2),
  sigma2_e_init = var(y_mh) * 0.5,
  sigma2_ah     = var(y_mh) * 0.5,
  mcmc_params   = mcmc,
  save_rds      = FALSE,
  verbose       = FALSE,
  map           = map_mh,
  windsize      = 5e6
)

cat("Top 10 MH windows by WPPA:\n")
top10_mh <- fit_mh$gwas[order(-fit_mh$gwas$WPPA), ][seq_len(min(10L, nrow(fit_mh$gwas))), ]
print(top10_mh, row.names = FALSE)

# ----- (C) Sanity-check QTL recovery on the demo ---------------------------
cat("\n== (C) QTL recovery sanity ==\n")
cat(sprintf(
  "  SNP path: median PIP at QTL = %.3f  vs  non-QTL = %.3f\n",
  median(fit_snp$pip[d$qtl$snp_idx]),
  median(fit_snp$pip[-d$qtl$snp_idx])
))
unit_index_mh <- match(attr(W_mh, "block_id"), map_mh$block_id)
qtl_blocks    <- unique(unit_index_mh[d$qtl$mh_idx])
cat(sprintf(
  "  MH path:  median pip_block at QTL = %.3f  vs  non-QTL = %.3f\n",
  median(fit_mh$pip_block[qtl_blocks]),
  median(fit_mh$pip_block[-qtl_blocks])
))

# ----- (D) Manhattan plots via CMplot --------------------------------------
# Plot -log10(1 - WPPA) / -log10(1 - PIP) per genomic window/marker with a
# posterior-probability threshold line. CMplot draws the chromosome bands
# and labels automatically. Cap probabilities at 1 - 1e-3 to keep
# -log10(0) = Inf out of the plot.

if (!requireNamespace("CMplot", quietly = TRUE)) {
  stop("Section (D) requires the CMplot package. Install with ",
       "install.packages(\"CMplot\").")
}

out_dir <- getwd()
wppa_cap <- 1 - 1e-3

# SNP path: per-window WPPA and per-SNP PIP.
snp_wppa_df <- data.frame(
  Wind = fit_snp$gwas$Wind,
  Chr  = fit_snp$gwas$Chr,
  Pos  = (fit_snp$gwas$Start + fit_snp$gwas$End) / 2,
  `1-WPPA` = 1 - pmin(fit_snp$gwas$WPPA, wppa_cap),
  check.names = FALSE
)
snp_pip_df <- data.frame(
  SNP = map_snp$SNP,
  Chr = map_snp$CHROM,
  Pos = map_snp$POS,
  `1-PIP` = 1 - pmin(fit_snp$pip, wppa_cap),
  check.names = FALSE
)

# MH path: per-window WPPA and per-block pip_block.
mh_wppa_df <- data.frame(
  Wind = fit_mh$gwas$Wind,
  Chr  = fit_mh$gwas$Chr,
  Pos  = (fit_mh$gwas$Start + fit_mh$gwas$End) / 2,
  `1-WPPA` = 1 - pmin(fit_mh$gwas$WPPA, wppa_cap),
  check.names = FALSE
)
# For MH block-level summary we use max per-allele PIP within each block
# instead of fit_mh$pip_block. The packaged pip_block is a union statistic
# (P(any allele in block has gamma != 0)), which inflates with block size
# k as approximately 1 - (1 - p)^k. On this demo (k = 3 alleles, per-allele
# noise floor ~0.4) the union floor sits at ~0.74, putting every non-QTL
# block above the 0.5 threshold and producing visually misleading FDR.
# Max-per-block has no such inflation (bounded by max(PIP) over alleles)
# and on the demo gives ~2.1x QTL vs non-QTL discrimination versus
# pip_block's ~1.3x, restoring the 0.5 threshold convention.
mh_allele_block <- attr(W_mh, "block_id")
max_pip_block <- vapply(map_mh$block_id, function(b) {
  max(fit_mh$pip[mh_allele_block == b])
}, numeric(1L))

mh_pip_df <- data.frame(
  Block = map_mh$block_id,
  Chr   = map_mh$chr,
  Pos   = (map_mh$start_pos + map_mh$end_pos) / 2,
  `1-PIP_max` = 1 - pmin(max_pip_block, wppa_cap),
  check.names = FALSE
)

# Highlight IDs whose posterior probability exceeds 0.5 (i.e. selects rows
# where `1 - WPPA < 0.5`). NULL is passed when nothing crosses the
# threshold so CMplot skips highlighting cleanly.
nonempty <- function(x) if (length(x) > 0L) as.character(x) else NULL
snp_wppa_hl <- nonempty(fit_snp$gwas$Wind[fit_snp$gwas$WPPA > 0.5])
snp_pip_hl  <- nonempty(map_snp$SNP[fit_snp$pip > 0.5])
mh_wppa_hl  <- nonempty(fit_mh$gwas$Wind[fit_mh$gwas$WPPA > 0.5])
mh_pip_hl   <- nonempty(map_mh$block_id[max_pip_block > 0.5])

cmplot_args <- list(
  type        = "h",
  plot.type   = "m",
  LOG10       = TRUE,
  threshold   = 0.5,
  threshold.lty = 2,
  threshold.lwd = 1,
  threshold.col = "red",
  amplify     = FALSE,
  highlight.col = NULL,
  file        = "jpg",
  dpi         = 150,
  file.output = TRUE,
  verbose     = FALSE
)

do.call(CMplot::CMplot, c(list(snp_wppa_df,
  ylab = expression(-log[10](1 - italic(WPPA))),
  file.name = "snp_wppa",
  highlight = snp_wppa_hl, highlight.text = snp_wppa_hl), cmplot_args))

do.call(CMplot::CMplot, c(list(snp_pip_df,
  ylab = expression(-log[10](1 - italic(PIP))),
  file.name = "snp_pip",
  highlight = snp_pip_hl, highlight.text = snp_pip_hl), cmplot_args))

do.call(CMplot::CMplot, c(list(mh_wppa_df,
  ylab = expression(-log[10](1 - italic(WPPA))),
  file.name = "mh_wppa",
  highlight = mh_wppa_hl, highlight.text = mh_wppa_hl), cmplot_args))

do.call(CMplot::CMplot, c(list(mh_pip_df,
  ylab = expression(-log[10](1 - italic(PIP)[max])),
  file.name = "mh_max_pip",
  highlight = mh_pip_hl, highlight.text = mh_pip_hl), cmplot_args))

cat(sprintf("\nManhattan plots written to: %s\n", out_dir))
cat("  Rect_Manhtn.{snp_wppa,snp_pip,mh_wppa,mh_max_pip}.jpg\n")

cat("\n04_gwas.R completed.\n")
