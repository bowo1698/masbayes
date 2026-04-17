// src/rust/src/matrix.rs

use ndarray::Array2;
use nalgebra::{DMatrix, SymmetricEigen};
use std::collections::HashMap;

/// Allele frequency 
#[derive(Debug, Clone)]
pub struct AlleleFreq {
    pub haplotype: String,
    pub allele: i32,
    pub freq: f64,
}

/// Reference structure for test set
#[derive(Debug, Clone)]
pub struct ReferenceStructure {
    pub allele_ids: Vec<String>,
    pub frequencies: Vec<f64>,
    pub basis_matrices: Vec<(String, Array2<f64>)>,  // (block_name, V) per block
}

/// Result structure
pub struct WMatrixResult {
    pub w_ah: Array2<f64>,
    pub allele_info: Vec<AlleleInfo>,
    pub basis_matrices: Vec<(String, Array2<f64>)>,
}

/// Allele metadata
#[derive(Debug, Clone)]
pub struct AlleleInfo {
    pub allele_id: String,
    pub block: String,
    pub allele: i32,
    pub freq: f64,
}

/// Convert ndarray Array2 to nalgebra DMatrix
fn to_nalgebra(arr: &Array2<f64>) -> DMatrix<f64> {
    let (nrow, ncol) = arr.dim();
    DMatrix::from_fn(nrow, ncol, |i, j| arr[[i, j]])
}

/// Compute eigen basis for W matrix of one block
/// Returns V: (h x h-1) matrix of eigenvectors with eigenvalue > threshold
fn compute_eigen_basis(w_block: &Array2<f64>) -> Array2<f64> {
    let n_ind = w_block.nrows();
    let h = w_block.ncols();

    if h <= 1 {
        return Array2::<f64>::eye(h);
    }

    // Compute W'W (h x h) — small matrix
    let wt = w_block.t();
    let mut wtw = Array2::<f64>::zeros((h, h));
    for i in 0..h {
        for j in 0..h {
            let mut sum = 0.0;
            for k in 0..n_ind {
                sum += wt[[i, k]] * w_block[[k, j]];
            }
            wtw[[i, j]] = sum;
        }
    }

    // Eigen decomposition via nalgebra
    let wtw_na = to_nalgebra(&wtw);
    let eigen = SymmetricEigen::new(wtw_na);

    // Keep h-1 eigenvectors with largest eigenvalues (drop smallest = null space)
    let threshold = 1e-8;
    let mut idx_vals: Vec<(usize, f64)> = eigen.eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    idx_vals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Take h-1 largest eigenvectors
    let keep: Vec<usize> = idx_vals.iter()
        .filter(|(_, v)| *v > threshold)
        .take(h - 1)
        .map(|(i, _)| *i)
        .collect();

    // Build V matrix (h x keep.len())
    if keep.is_empty() {
        // Fallback: return identity-like single column
        let mut v = Array2::<f64>::zeros((h, 1));
        v[[0, 0]] = 1.0;
        return v;
    }
    let n_keep = keep.len();
    let mut v = Array2::<f64>::zeros((h, n_keep));
    for (new_j, &old_j) in keep.iter().enumerate() {
        for i in 0..h {
            v[[i, new_j]] = eigen.eigenvectors[(i, old_j)];
        }
    }

    v
}

/// Project W_block (n x h) using basis V (h x h-1) -> W_reduced (n x h-1)
fn project_with_basis(w_block: &Array2<f64>, v: &Array2<f64>) -> Array2<f64> {
    let n_ind = w_block.nrows();
    let n_basis = v.ncols();
    let h = v.nrows();

    let mut w_reduced = Array2::<f64>::zeros((n_ind, n_basis));
    for i in 0..n_ind {
        for j in 0..n_basis {
            let mut sum = 0.0;
            for k in 0..h {
                sum += w_block[[i, k]] * v[[k, j]];
            }
            w_reduced[[i, j]] = sum;
        }
    }
    w_reduced
}

/// Main W matrix builder
pub struct WMatrixBuilder {
    hap_matrix: Array2<i32>,
    n_individuals: usize,
    n_blocks: usize,
    colnames: Vec<String>,
    allele_freq_map: HashMap<String, HashMap<i32, f64>>,
}

impl WMatrixBuilder {
    pub fn new(
        hap_matrix: Array2<i32>,
        colnames: Vec<String>,
        allele_freq: Vec<AlleleFreq>,
    ) -> Self {
        let n_individuals = hap_matrix.nrows();
        let n_blocks = hap_matrix.ncols() / 2;

        let mut freq_map: HashMap<String, HashMap<i32, f64>> = HashMap::new();
        for af in allele_freq {
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
        }
    }

    pub fn get_block_base_name(haplotype: &str) -> String {
        haplotype
            .strip_suffix("_1")
            .unwrap_or(haplotype)
            .to_string()
    }

    pub fn build(&self) -> WMatrixResult {
        let mut w_reduced_blocks: Vec<Array2<f64>> = Vec::new();
        let mut all_allele_info: Vec<AlleleInfo> = Vec::new();
        let mut all_basis: Vec<(String, Array2<f64>)> = Vec::new();

        for block_idx in 0..self.n_blocks {
            let col1 = 2 * block_idx;
            let col2 = 2 * block_idx + 1;

            let block_name1 = &self.colnames[col1];
            let block_base = Self::get_block_base_name(block_name1);

            let freq_block = match self.allele_freq_map.get(&block_base) {
                Some(freqs) => freqs,
                None => continue,
            };

            if freq_block.is_empty() { continue; }

            let allele1 = self.hap_matrix.column(col1);
            let allele2 = self.hap_matrix.column(col2);

            // Sort all alleles by frequency descending
            let mut freq_vec: Vec<(i32, f64)> = freq_block.iter()
                .map(|(k, v)| (*k, *v))
                .collect();
            freq_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            if freq_vec.is_empty() { continue; }

            let h = freq_vec.len();

            // Build full W_block (n x h) — semua h allele, tidak ada yang di-drop
            let mut w_block = Array2::<f64>::zeros((self.n_individuals, h));

            for (k_idx, (allele_k, freq_k)) in freq_vec.iter().enumerate() {
                let allele_k = *allele_k;
                let p_k = *freq_k;

                for i in 0..self.n_individuals {
                    let a1 = allele1[i];
                    let a2 = allele2[i];

                    let value = if a1 == allele_k && a2 == allele_k {
                        -2.0 * (1.0 - p_k)
                    } else if (a1 == allele_k && a2 != allele_k) ||
                              (a1 != allele_k && a2 == allele_k) {
                        -(1.0 - 2.0 * p_k)
                    } else {
                        2.0 * p_k
                    };

                    w_block[[i, k_idx]] = value;
                }

                // Metadata untuk semua h allele
                all_allele_info.push(AlleleInfo {
                    allele_id: format!("{}_allele{}", block_name1, allele_k),
                    block: block_name1.clone(),
                    allele: allele_k,
                    freq: p_k,
                });
            }

            if h == 1 {
                // Single allele block — tidak perlu eigen decomp
                all_basis.push((block_name1.clone(), Array2::<f64>::eye(1)));
                w_reduced_blocks.push(w_block);
                continue;
            }

            // Eigen decomposition -> basis V (h x h-1)
            let v = compute_eigen_basis(&w_block);

            // Project: W_reduced = W_block * V  (n x h-1)
            let w_reduced = project_with_basis(&w_block, &v);

            all_basis.push((block_name1.clone(), v));
            w_reduced_blocks.push(w_reduced);
        }

        // Combine semua reduced blocks
        let total_cols: usize = w_reduced_blocks.iter().map(|b| b.ncols()).sum();
        let mut w_ah = Array2::<f64>::zeros((self.n_individuals, total_cols));

        let mut col_offset = 0;
        for block in &w_reduced_blocks {
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
            basis_matrices: all_basis,
        }
    }

    /// Build W matrix for test set using reference basis from training
    pub fn build_with_reference(
        hap_matrix: Array2<i32>,
        colnames: Vec<String>,
        reference: &ReferenceStructure,
    ) -> Array2<f64> {
        let n_individuals = hap_matrix.nrows();
        let n_blocks_hap  = hap_matrix.ncols() / 2;

        // Group allele_ids per block — simpan urutan allele sesuai allele_ids
        let mut block_alleles: HashMap<String, Vec<(usize, i32, f64)>> = HashMap::new();
        for (col_idx, allele_id) in reference.allele_ids.iter().enumerate() {
            if let Some(pos) = allele_id.rfind("_allele") {
                let block_name  = allele_id[..pos].to_string();
                let allele_num: i32 = allele_id[pos+7..].parse().unwrap_or(0);
                let p_k = reference.frequencies[col_idx];
                block_alleles.entry(block_name).or_default().push((col_idx, allele_num, p_k));
            }
        }

        // Total output cols = sum of basis ncols per block (= h-1 per block)
        let total_cols: usize = reference.basis_matrices.iter()
            .map(|(_, v)| v.ncols())
            .sum();
        let mut w_ah = Array2::<f64>::zeros((n_individuals, total_cols));

        let mut col_offset = 0;
        for (block_name, v) in &reference.basis_matrices {
            let alleles = match block_alleles.get(block_name) {
                Some(a) => a,
                None => {
                    col_offset += v.ncols();
                    continue;
                }
            };

            // Find haplotype columns for this block
            let mut hap_idx: Option<(usize, usize)> = None;
            for block_idx in 0..n_blocks_hap {
                let col1 = 2 * block_idx;
                let col2 = 2 * block_idx + 1;
                let base = WMatrixBuilder::get_block_base_name(&colnames[col1]);
                if colnames[col1] == *block_name || base == *block_name {
                    hap_idx = Some((col1, col2));
                    break;
                }
            }

            let (col1, col2) = match hap_idx {
                Some(idx) => idx,
                None => {
                    col_offset += v.ncols();
                    continue;
                }
            };

            let allele1 = hap_matrix.column(col1);
            let allele2 = hap_matrix.column(col2);
            let h = alleles.len();

            // Build W_block_test (n_test x h) — semua h allele
            let mut w_block = Array2::<f64>::zeros((n_individuals, h));
            for (k_idx, (_, allele_num, p_k)) in alleles.iter().enumerate() {
                for i in 0..n_individuals {
                    let a1 = allele1[i];
                    let a2 = allele2[i];
                    let value = if a1 == *allele_num && a2 == *allele_num {
                        -2.0 * (1.0 - p_k)
                    } else if (a1 == *allele_num && a2 != *allele_num) ||
                            (a1 != *allele_num && a2 == *allele_num) {
                        -(1.0 - 2.0 * p_k)
                    } else {
                        2.0 * p_k
                    };
                    w_block[[i, k_idx]] = value;
                }
            }

            // Project dengan basis V dari training: W_reduced = W_block * V
            let w_reduced = project_with_basis(&w_block, v);
            let n_reduced = w_reduced.ncols();

            for i in 0..n_individuals {
                for j in 0..n_reduced {
                    w_ah[[i, col_offset + j]] = w_reduced[[i, j]];
                }
            }

            col_offset += n_reduced;
        }

        w_ah
    }
}