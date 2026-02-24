//! Spectral Projector: Johnson-Lindenstrauss random projection for O(n²) matmul.
//!
//! Equation: Y = (1/√D) · R · X, where R ∈ {-1, +1}^{D×n}
//!
//! Uses contiguous f64 buffers for cache-friendly access.
//! Projection weights are generated deterministically via Feistel for reproducibility.

use crate::feistel::FeistelMemoizer;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

/// Contiguous row-major matrix.
pub struct DenseMatrix {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl DenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    /// From a Python list-of-lists.
    pub fn from_lol(rows: &[Vec<f64>]) -> Self {
        let r = rows.len();
        let c = if r > 0 { rows[0].len() } else { 0 };
        let mut data = Vec::with_capacity(r * c);
        for row in rows {
            data.extend_from_slice(row);
        }
        Self {
            data,
            rows: r,
            cols: c,
        }
    }

    /// To a nested Vec for Python return.
    pub fn to_lol(&self) -> Vec<Vec<f64>> {
        self.data.chunks(self.cols).map(|c| c.to_vec()).collect()
    }
}

/// Spectral Projector using JL random projection.
pub struct SpectralProjector {
    feistel: FeistelMemoizer,
    target_dim: usize,
    seed: u64,
}

impl SpectralProjector {
    pub fn new(target_dim: usize, seed: u64) -> Self {
        Self {
            feistel: FeistelMemoizer::new(4),
            target_dim,
            seed,
        }
    }

    /// Generate a deterministic pseudo-random weight in [-1, 1].
    #[inline]
    fn weight(&self, row: usize, col: usize) -> f64 {
        let input = ((self.seed as u128) << 64) | ((row as u128) << 32) | (col as u128);
        let projected = self.feistel.project_to_seed(input);
        // Map u128 → [-1.0, 1.0]
        (projected as u64 as f64) / (u64::MAX as f64) * 2.0 - 1.0
    }

    /// Project matrix A (m×k) down to (m×D) where D = target_dim.
    pub fn project(&self, a: &DenseMatrix) -> DenseMatrix {
        let m = a.rows;
        let k = a.cols;
        let d = self.target_dim;
        let scale = 1.0 / (d as f64).sqrt();

        let mut result = DenseMatrix::new(m, d);

        // Parallelized across rows
        result.data = (0..m)
            .into_par_iter()
            .flat_map(|i| {
                let mut row = vec![0.0f64; d];
                for j in 0..d {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += a.get(i, l) * self.weight(j, l);
                    }
                    row[j] = sum * scale;
                }
                row
            })
            .collect();

        result
    }

    /// O(n²) spectral matmul: project both A and B, then multiply projections.
    pub fn spectral_matmul(&self, a: &DenseMatrix, b: &DenseMatrix) -> DenseMatrix {
        // Project A (m×k) → (m×D) and B (k×n) → (D×n) via B^T projection
        let a_proj = self.project(a);

        // For B, we project its transpose
        let bt = transpose(b);
        let bt_proj = self.project(&bt);
        let b_proj = transpose(&bt_proj);

        // Now multiply a_proj (m×D) × b_proj (D×n)
        naive_matmul(&a_proj, &b_proj)
    }
}

fn transpose(m: &DenseMatrix) -> DenseMatrix {
    let mut t = DenseMatrix::new(m.cols, m.rows);
    for i in 0..m.rows {
        for j in 0..m.cols {
            t.set(j, i, m.get(i, j));
        }
    }
    t
}

/// Simple O(n³) matmul for the reduced-dimension projected matrices.
fn naive_matmul(a: &DenseMatrix, b: &DenseMatrix) -> DenseMatrix {
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    let mut result = DenseMatrix::new(m, n);

    // Parallelize across rows
    result.data = (0..m)
        .into_par_iter()
        .flat_map(|i| {
            let mut row = vec![0.0f64; n];
            for l in 0..k {
                let a_il = a.get(i, l);
                for j in 0..n {
                    row[j] += a_il * b.get(l, j);
                }
            }
            row
        })
        .collect();

    result
}

// --- PyO3 Wrapper ---

#[pyclass]
pub struct PySpectralProjector {
    inner: SpectralProjector,
}

#[pymethods]
impl PySpectralProjector {
    #[new]
    #[pyo3(signature = (target_dim=128, seed=42))]
    fn new(target_dim: usize, seed: u64) -> Self {
        Self {
            inner: SpectralProjector::new(target_dim, seed),
        }
    }

    /// Spectral matmul: takes two list-of-lists, returns list-of-lists.
    fn matmul(&self, a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let dm_a = DenseMatrix::from_lol(&a);
        let dm_b = DenseMatrix::from_lol(&b);
        let result = self.inner.spectral_matmul(&dm_a, &dm_b);
        result.to_lol()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_shapes() {
        let sp = SpectralProjector::new(32, 42);
        let a = DenseMatrix::new(10, 20);
        let proj = sp.project(&a);
        assert_eq!(proj.rows, 10);
        assert_eq!(proj.cols, 32);
    }

    #[test]
    fn test_spectral_matmul_shape() {
        let sp = SpectralProjector::new(16, 42);
        let a = DenseMatrix::new(5, 8);
        let b = DenseMatrix::new(8, 6);
        let result = sp.spectral_matmul(&a, &b);
        assert_eq!(result.rows, 5);
        assert_eq!(result.cols, 6);
    }
}
