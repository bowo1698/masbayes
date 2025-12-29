// src/rust/src/matrix.rs

use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};
use extendr_api::prelude::*;

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
        // Remove common suffixes like "_copy", "_2", etc.
        haplotype
            .trim_end_matches("_copy")
            .trim_end_matches(|c: char| c.is_digit(10) && haplotype.ends_with(&c.to_string()))
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
            let block_name2 = &self.colnames[col2];
            
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
    
    /// Build W matrix for test set using reference structure
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
        
        w_ah
    }
}