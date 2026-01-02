// src/rust/src/variance_tuning.rs

use ndarray::{Array1, Array2};

/// Estimate genetic variance using GBLUP-style approach
pub fn estimate_genetic_variance(
    w: &Array2<f64>,
    y: &Array1<f64>,
) -> f64 {
    let n = w.nrows();
    let p = w.ncols();
    
    // Compute genomic relationship matrix G = WW'/p
    let wt = w.t();
    let g = w.dot(&wt) / (p as f64);
    
    // Simple variance component estimation via method of moments
    // Var(y) = σ²_g + σ²_e
    // E[y'Gy/n] ≈ σ²_g
    
    let y_mean = y.mean().unwrap();
    let mut y_centered = y.clone();
    for val in y_centered.iter_mut() {
        *val -= y_mean;
    }
    
    let var_total = y_centered.iter().map(|v| v.powi(2)).sum::<f64>() / (n as f64 - 1.0);
    
    // Quick estimate: σ²_g ≈ y'Gy/n
    let gy = g.dot(&y_centered);
    let ytgy = y_centered.iter().zip(gy.iter()).map(|(a, b)| a * b).sum::<f64>();
    let sigma2_g = (ytgy / n as f64).max(0.0);
    
    // Bound to reasonable range
    let sigma2_g_bounded = sigma2_g.min(var_total * 0.9).max(var_total * 0.1);
    
    sigma2_g_bounded
}

/// Generate adaptive variance grid based on genetic variance
pub fn adaptive_variance_grid(sigma2_g: f64) -> [f64; 4] {
    // Component 0: zero (spike at origin)
    // Components 1-3: scaled by genetic variance
    // Using geometric spacing for better coverage
    
    [
        0.0,                    // Zero component
        sigma2_g * 0.0001,      // Small effects (0.01% of σ²_g)
        sigma2_g * 0.001,       // Medium effects (0.1% of σ²_g)
        sigma2_g * 0.01,        // Large effects (1% of σ²_g)
    ]
}

/// Alternative: More aggressive grid for oligogenic traits
pub fn adaptive_variance_grid_sparse(sigma2_g: f64) -> [f64; 4] {
    [
        0.0,
        sigma2_g * 0.00001,     // Very small
        sigma2_g * 0.001,       // Medium
        sigma2_g * 0.1,         // Very large (10% of σ²_g)
    ]
}

/// Estimate effective number of QTL to decide grid type
pub fn estimate_architecture_type(
    w: &Array2<f64>,
    y: &Array1<f64>,
) -> bool {
    // Simple heuristic: compute variance explained by top markers
    // If top 1% markers explain >50% variance → oligogenic (true)
    // Otherwise → polygenic (false)
    
    let n = w.nrows();
    let p = w.ncols();
    
    // Compute marginal correlations
    let y_mean = y.mean().unwrap();
    let mut y_centered = y.clone();
    for val in y_centered.iter_mut() {
        *val -= y_mean;
    }
    
    let y_var = y_centered.iter().map(|v| v.powi(2)).sum::<f64>() / (n as f64 - 1.0);
    
    let mut cors: Vec<f64> = Vec::with_capacity(p);
    for j in 0..p {
        let w_col = w.column(j);
        let w_mean = w_col.mean().unwrap();
        
        let mut cov = 0.0;
        let mut w_var = 0.0;
        for i in 0..n {
            let w_centered = w_col[i] - w_mean;
            cov += w_centered * y_centered[i];
            w_var += w_centered.powi(2);
        }
        
        let cor = if w_var > 1e-10 {
            (cov / (w_var * y_var).sqrt()).abs()
        } else {
            0.0
        };
        cors.push(cor);
    }
    
    // Sort correlations
    cors.sort_by(|a, b| b.partial_cmp(a).unwrap());
    
    // Check top 1%
    let top_1pct = (p as f64 * 0.01).max(10.0) as usize;
    let top_1pct = top_1pct.min(cors.len());
    
    let variance_top = cors.iter().take(top_1pct).map(|c| c.powi(2)).sum::<f64>();
    
    // If top 1% explains >30% variance → sparse/oligogenic
    variance_top > 0.3
}