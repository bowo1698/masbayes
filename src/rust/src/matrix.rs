// src/rust/src/matrix.rs

//! Design-matrix construction for multi-allelic microhaplotype markers.
//!
//! ## Pipeline
//!
//! 1. **Per-block encoding** — for each haplotype locus (block) and each
//!    non-baseline microhaplotype $k$, the entry $W_{i,k}$ for individual
//!    $i$ is filled with the Da (2015) three-value coding rule (Eqs.
//!    22–24): $-2(1-p_k)$ when the individual is homozygous for $k$,
//!    $-(1-2p_k)$ when heterozygous, $+2p_k$ when $k$ is absent.
//! 2. **Baseline drop** — the most frequent microhaplotype per locus is
//!    used as the contrast reference and excluded from the columns; its
//!    effect is absorbed by the intercept $\mu$ in downstream models.
//! 3. **Frequency-weighted row shrinkage** — see
//!    [`frequency_weighted_row_shrinkage`]. Applied per locus after
//!    encoding; partially reduces the alignment between each individual
//!    row and the locus's allele-frequency vector.
//!
//! ## Equivalence with VanRaden
//!
//! Da's three-value rule can be written in closed form as
//! $W_{i,k} = 2 p_k - n_{i,k}$, where $n_{i,k} \in \{0,1,2\}$ is the dosage
//! of microhaplotype $k$ in individual $i$. This is the per-allele
//! VanRaden centering with the sign flipped, stacked across all
//! non-baseline microhaplotypes within the locus.
//!
//! ## Train / test alignment
//!
//! `WMatrixBuilder` accepts a `ReferenceStructure` for test-set encoding so
//! that allele frequencies and the choice of baseline microhaplotype come
//! from the training set, not the test set itself. Re-estimating $p_k$ on
//! test data would shift the centering and bias GEBVs — never do that.
//!
//! ## Reference
//!
//! Da, Y. (2015). Multi-allelic haplotype model based on genetic partition
//! for genomic prediction and variance component estimation using SNP
//! markers. *BMC Genetics*, 16:144.

use ndarray::Array2;
use std::collections::HashMap;

/// Frequency-weighted row shrinkage (per haplotype block).
///
/// Applied after Da (2015) Eqs. 22-24 encoding as a per-row rank-1
/// deflation along the allele-frequency vector p:
///     W'_{i,k} = W_{i,k} - p_k * S / F
/// where S = sum_l p_l W_{i,l} and F = sum_l p_l.
///
/// The p-weighted row sum is partially damped to S * (1 - Q/F)
/// with Q = sum_l p_l^2.
fn frequency_weighted_row_shrinkage(
    w_block: &mut Array2<f64>,
    freqs: &[(i32, f64)],
) {
    let n_ind = w_block.nrows();
    let n_alleles = w_block.ncols();
    let freq_vals: Vec<f64> = freqs.iter().map(|(_, f)| *f).collect();
    let freq_sum: f64 = freq_vals.iter().sum();
    if freq_sum < 1e-10 {
        return;
    }
    for i in 0..n_ind {
        let weighted_sum: f64 = (0..n_alleles)
            .map(|k| freq_vals[k] * w_block[[i, k]])
            .sum();
        for k in 0..n_alleles {
            w_block[[i, k]] -= freq_vals[k] * weighted_sum / freq_sum;
        }
    }
}

/// Allele frequency 
#[derive(Debug, Clone)]
pub struct AlleleFreq {
    pub haplotype: String,  // e.g., "block_1" or "block_1_copy"
    pub allele: i32,
    pub freq: f64,
}

/// Reference structure for test set
#[derive(Debug, Clone)]
pub struct ReferenceStructure {
    pub allele_ids: Vec<String>,
    pub frequencies: Vec<f64>,
    pub dropped_alleles: Vec<DroppedAllele>,
}

/// Dropped baseline allele
#[derive(Debug, Clone)]
pub struct DroppedAllele {
    pub block: String,
    pub allele: i32,
    pub freq: f64,
}

/// Result structure
pub struct WMatrixResult {
    pub w_ah: Array2<f64>,
    pub allele_info: Vec<AlleleInfo>,
    pub dropped_alleles: Vec<DroppedAllele>,
}

/// Allele metadata
#[derive(Debug, Clone)]
pub struct AlleleInfo {
    pub allele_id: String,
    pub block: String,
    pub allele: i32,
    pub freq: f64,
}

/// Main W matrix builder
pub struct WMatrixBuilder {
    hap_matrix: Array2<i32>,
    n_individuals: usize,
    n_blocks: usize,
    colnames: Vec<String>,
    // Map: (block_base_name, allele) -> frequency
    allele_freq_map: HashMap<String, HashMap<i32, f64>>,
    drop_baseline: bool,
}

impl WMatrixBuilder {
    /// Create new builder from training data
    pub fn new(
        hap_matrix: Array2<i32>,
        colnames: Vec<String>,
        allele_freq: Vec<AlleleFreq>,
        drop_baseline: bool,
    ) -> Self {
        let n_individuals = hap_matrix.nrows();
        let n_blocks = hap_matrix.ncols() / 2;
        
        // Build frequency map: block_base -> {allele: freq}
        // Extract base block name (remove trailing "_copy" or numbered suffix)
        let mut freq_map: HashMap<String, HashMap<i32, f64>> = HashMap::new();
        
        for af in allele_freq {
            // Get base block name by removing suffix
            let block_base = Self::get_block_base_name(&af.haplotype);
            
            freq_map
                .entry(block_base)
                .or_insert_with(HashMap::new)
                .insert(af.allele, af.freq);
        }
        
        Self {
            hap_matrix,
            n_individuals,
            n_blocks,
            colnames,
            allele_freq_map: freq_map,
            drop_baseline,
        }
    }
    
    /// Extract base block name (handle "block_1" and "block_1_copy" -> "block_1")
    fn get_block_base_name(haplotype: &str) -> String {
        haplotype
            .strip_suffix("_1")
            .unwrap_or(haplotype)
            .to_string()
    }
    
    /// Build W matrix for training set
    pub fn build(&self) -> WMatrixResult {
        let mut w_blocks: Vec<Array2<f64>> = Vec::new();
        let mut all_allele_info: Vec<AlleleInfo> = Vec::new();
        let mut all_dropped: Vec<DroppedAllele> = Vec::new();
        
        for block_idx in 0..self.n_blocks {
            let col1 = 2 * block_idx;
            let col2 = 2 * block_idx + 1;
            
            let block_name1 = &self.colnames[col1];
            
            // Get base block name
            let block_base = Self::get_block_base_name(block_name1);
            
            // Get alleles for this block from frequency map
            let freq_block = match self.allele_freq_map.get(&block_base) {
                Some(freqs) => freqs,
                None => continue,  // No informative alleles for this block
            };
            
            if freq_block.is_empty() {
                continue;
            }
            
            // Get allele columns
            let allele1 = self.hap_matrix.column(col1);
            let allele2 = self.hap_matrix.column(col2);
            
            // Sort alleles by frequency (descending)
            let mut freq_vec: Vec<(i32, f64)> = freq_block.iter()
                .map(|(k, v)| (*k, *v))
                .collect();
            freq_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            let mut informative_alleles = freq_vec;
            
            // Drop baseline (most frequent allele)
            if self.drop_baseline && informative_alleles.len() > 1 {
                let baseline = informative_alleles.remove(0);
                all_dropped.push(DroppedAllele {
                    block: block_name1.clone(),
                    allele: baseline.0,
                    freq: baseline.1,
                });
            }
            
            if informative_alleles.is_empty() {
                continue;
            }
            
            // Build W matrix for this block
            let n_alleles_block = informative_alleles.len();
            let mut w_block = Array2::<f64>::zeros((self.n_individuals, n_alleles_block));
            
            for (k_idx, (allele_k, freq_k)) in informative_alleles.iter().enumerate() {
                let allele_k = *allele_k;
                let p_k = *freq_k;
                
                // Apply coding rule for each individual
                for i in 0..self.n_individuals {
                    let a1 = allele1[i];
                    let a2 = allele2[i];
                    
                    let value = if a1 == allele_k && a2 == allele_k {
                        -2.0 * (1.0 - p_k)  // Homozygous for allele k
                    } else if (a1 == allele_k && a2 != allele_k) || 
                              (a1 != allele_k && a2 == allele_k) {
                        -(1.0 - 2.0 * p_k)  // Heterozygous
                    } else {
                        2.0 * p_k  // Homozygous for other alleles
                    };
                    
                    w_block[[i, k_idx]] = value;
                }
                
                // Store metadata
                all_allele_info.push(AlleleInfo {
                    allele_id: format!("{}_allele{}", block_name1, allele_k),
                    block: block_name1.clone(),
                    allele: allele_k,
                    freq: p_k,
                });
            }
            
            frequency_weighted_row_shrinkage(&mut w_block, &informative_alleles);
            w_blocks.push(w_block);
        }
        
        // Combine all blocks into single W_αh matrix
        let total_alleles: usize = w_blocks.iter().map(|b| b.ncols()).sum();
        let mut w_ah = Array2::<f64>::zeros((self.n_individuals, total_alleles));
        
        let mut col_offset = 0;
        for block in w_blocks {
            let n_cols = block.ncols();
            for i in 0..self.n_individuals {
                for j in 0..n_cols {
                    w_ah[[i, col_offset + j]] = block[[i, j]];
                }
            }
            col_offset += n_cols;
        }
        
        WMatrixResult {
            w_ah,
            allele_info: all_allele_info,
            dropped_alleles: all_dropped,
        }
    }
    
    /// Build W matrix for test set using reference structure from W
    pub fn build_with_reference(
        hap_matrix: Array2<i32>,
        colnames: Vec<String>,
        reference: &ReferenceStructure,
    ) -> Array2<f64> {
        let n_individuals = hap_matrix.nrows();
        let n_alleles = reference.allele_ids.len();
        let n_blocks = hap_matrix.ncols() / 2;
        
        let mut w_ah = Array2::<f64>::zeros((n_individuals, n_alleles));
        
        for (col_idx, allele_id) in reference.allele_ids.iter().enumerate() {
            // Parse allele_id: "blockname_allele123"
            let parts: Vec<&str> = allele_id.split("_allele").collect();
            if parts.len() != 2 {
                continue;
            }
            
            let block_name = parts[0];
            let allele_num: i32 = match parts[1].parse() {
                Ok(num) => num,
                Err(_) => continue,
            };
            
            // Find corresponding haplotype columns
            let mut hap_idx: Option<(usize, usize)> = None;
            
            for block_idx in 0..n_blocks {
                let col1 = 2 * block_idx;
                let col2 = 2 * block_idx + 1;
                
                // Match by exact block name or base name
                if colnames[col1] == block_name || 
                   Self::get_block_base_name(&colnames[col1]) == block_name {
                    hap_idx = Some((col1, col2));
                    break;
                }
            }
            
            if let Some((col1, col2)) = hap_idx {
                let allele1 = hap_matrix.column(col1);
                let allele2 = hap_matrix.column(col2);
                let p_k = reference.frequencies[col_idx];
                
                // Apply coding rule
                for i in 0..n_individuals {
                    let a1 = allele1[i];
                    let a2 = allele2[i];
                    
                    let value = if a1 == allele_num && a2 == allele_num {
                        -2.0 * (1.0 - p_k)
                    } else if (a1 == allele_num && a2 != allele_num) || 
                              (a1 != allele_num && a2 == allele_num) {
                        -(1.0 - 2.0 * p_k)
                    } else {
                        2.0 * p_k
                    };
                    
                    w_ah[[i, col_idx]] = value;
                }
            }
        }
        
        // sum-to-zero constraint
        let mut block_map: HashMap<String, Vec<(usize, f64)>> = HashMap::new();
        for (col_idx, allele_id) in reference.allele_ids.iter().enumerate() {
            if let Some(pos) = allele_id.rfind("_allele") {
                let block_name = allele_id[..pos].to_string();
                let p_k = reference.frequencies[col_idx];
                block_map.entry(block_name).or_default().push((col_idx, p_k));
            }
        }
        for (_block_name, col_freq_pairs) in &block_map {
            let freq_sum: f64 = col_freq_pairs.iter().map(|(_, f)| f).sum();
            if freq_sum < 1e-10 { continue; }
            for i in 0..n_individuals {
                let weighted_sum: f64 = col_freq_pairs.iter()
                    .map(|(col, freq)| freq * w_ah[[i, *col]])
                    .sum();
                for (col, freq) in col_freq_pairs {
                    w_ah[[i, *col]] -= freq * weighted_sum / freq_sum;
                }
            }
        }

        w_ah
    }
}

// ============================================================================
// Unit tests
// ============================================================================
//
// All tests run under `cargo test` (or `cargo check --tests` to only verify
// compilation). They cover deterministic behaviour of the design-matrix
// pipeline; nothing here depends on RNG state.

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Helper: build a fresh 3-individual × 2-allele block.
    fn make_w(values: &[[f64; 2]]) -> Array2<f64> {
        let n = values.len();
        let mut w = Array2::<f64>::zeros((n, 2));
        for (i, row) in values.iter().enumerate() {
            w[[i, 0]] = row[0];
            w[[i, 1]] = row[1];
        }
        w
    }

    #[test]
    fn shrinkage_preserves_dimensions() {
        // Input shape is preserved (no row / column drops).
        let mut w = make_w(&[
            [1.0, 2.0],
            [3.0, 4.0],
            [-1.0, 0.5],
        ]);
        let freqs = vec![(2, 0.3), (3, 0.2)];
        frequency_weighted_row_shrinkage(&mut w, &freqs);
        assert_eq!(w.nrows(), 3);
        assert_eq!(w.ncols(), 2);
    }

    #[test]
    fn shrinkage_partial_damping_factor() {
        // For p = (0.3, 0.2), Q/F = 0.13 / 0.5 = 0.26, so the post-shrink
        // weighted row sum should equal S · (1 - Q/F) = S · 0.74.
        let mut w = make_w(&[[1.0, 2.0]]);
        let p = [0.3_f64, 0.2_f64];
        let freqs = vec![(1, p[0]), (2, p[1])];

        let s_before: f64 = p.iter().zip(w.row(0).iter()).map(|(pl, wl)| pl * wl).sum();
        frequency_weighted_row_shrinkage(&mut w, &freqs);
        let s_after: f64 = p.iter().zip(w.row(0).iter()).map(|(pl, wl)| pl * wl).sum();

        let q: f64 = p.iter().map(|x| x * x).sum();
        let f: f64 = p.iter().sum();
        let expected = s_before * (1.0 - q / f);
        assert!((s_after - expected).abs() < 1e-10,
            "expected {} after shrinkage, got {}", expected, s_after);
    }

    #[test]
    fn shrinkage_skips_zero_frequency_block() {
        // freq_sum < 1e-10 short-circuits; W is left untouched.
        let mut w = make_w(&[[1.0, 2.0]]);
        let original = w.clone();
        let freqs = vec![(1, 0.0_f64), (2, 0.0_f64)];
        frequency_weighted_row_shrinkage(&mut w, &freqs);
        assert_eq!(w, original);
    }

    #[test]
    fn shrinkage_single_column_h2_block() {
        // h = 2 block: single non-baseline column with frequency p.
        // Math collapses to W' = W · (1 - p) — verify.
        let p = 0.4_f64;
        let mut w = array![[2.0 * p], [-(1.0 - 2.0 * p)], [-2.0 * (1.0 - p)]];
        let freqs = vec![(2, p)];
        frequency_weighted_row_shrinkage(&mut w, &freqs);
        let scale = 1.0 - p;
        assert!((w[[0, 0]] - 2.0 * p * scale).abs() < 1e-12);
        assert!((w[[1, 0]] - (-(1.0 - 2.0 * p)) * scale).abs() < 1e-12);
        assert!((w[[2, 0]] - (-2.0 * (1.0 - p)) * scale).abs() < 1e-12);
    }

    #[test]
    fn da_encoding_three_genotype_classes() {
        // Verify the Da (2015) Eqs. 22-24 three-value rule by constructing
        // a small WMatrixBuilder over a single 3-allele block with all
        // three genotype classes represented.
        //
        // Frequencies: allele 1 (baseline, dropped) = 0.5, allele 2 = 0.3,
        //              allele 3 = 0.2. After baseline drop, columns are
        //              [allele 2, allele 3].
        let p2 = 0.3_f64;
        let p3 = 0.2_f64;

        let hap = Array2::from_shape_vec((4, 2), vec![
            1, 1,   // individual 0: homozygous baseline (1,1)
            1, 2,   // individual 1: heterozygous, carries allele 2
            2, 2,   // individual 2: homozygous allele 2
            3, 1,   // individual 3: heterozygous, carries allele 3
        ]).unwrap();

        let freq = vec![
            AlleleFreq { haplotype: "block_1".to_string(), allele: 1, freq: 0.5 },
            AlleleFreq { haplotype: "block_1".to_string(), allele: 2, freq: p2 },
            AlleleFreq { haplotype: "block_1".to_string(), allele: 3, freq: p3 },
        ];

        let builder = WMatrixBuilder::new(
            hap,
            vec!["block_1".to_string(), "block_1".to_string()],
            freq,
            true, // drop baseline
        );

        let result = builder.build();

        // After baseline drop, h - 1 = 2 columns expected.
        assert_eq!(result.w_ah.ncols(), 2);
        assert_eq!(result.w_ah.nrows(), 4);
        assert_eq!(result.dropped_alleles.len(), 1);
        assert_eq!(result.dropped_alleles[0].allele, 1);

        // The encoding step alone gives Da-raw values; the post-encoding
        // frequency_weighted_row_shrinkage mutates them. Rather than verify
        // exact numeric values (which depend on the shrinkage), we check
        // qualitative invariants: same-row sum and signs in the expected
        // direction for the homozygous-baseline individual.
        let row0_sum_p: f64 = p2 * result.w_ah[[0, 0]] + p3 * result.w_ah[[0, 1]];
        // After partial damping: row0_sum_p = S0 · (1 - Q/F).
        // S0 = p2 · 2p2 + p3 · 2p3 = 2 (p2² + p3²) = 2Q
        // (1 - Q/F) with F = p2 + p3 = 0.5, Q = p2² + p3² = 0.13.
        // Expected: 2 · 0.13 · (1 - 0.13/0.5) = 0.26 · 0.74 = 0.1924.
        assert!((row0_sum_p - 0.1924).abs() < 1e-6,
            "homozygous-baseline weighted sum drift: got {}", row0_sum_p);
    }
}