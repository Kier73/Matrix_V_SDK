use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

/// T-Matrix Engine: Native acceleration for Morphological and Topological matrices.
///
/// MATHEMATICAL FOUNDATION:
/// 1. Gielis Superformula (Morphological DNA):
///    r(phi) = [ |cos(m*phi/4)/a|^n2 + |sin(m*phi/4)/b|^n3 ]^(-1/n1)
///
/// 2. HDC/Holographic Mapping (Topological Variety):
///    phi_{i,j} = Hash2D(i, j) -> [0, 2*PI]
///    W_{i,j} = r(phi_{i,j})
#[pyclass]
pub struct PyTMatrixEngine;

#[pymethods]
impl PyTMatrixEngine {
    #[new]
    pub fn new() -> Self {
        PyTMatrixEngine
    }

    /// Native Gielis Superformula Implementation.
    #[pyo3(text_signature = "(phi, m, a, b, n1, n2, n3)")]
    pub fn r_gielis(
        &self,
        phi: Vec<f32>,
        m: f32,
        a: f32,
        b: f32,
        n1: f32,
        n2: f32,
        n3: f32,
    ) -> Vec<f32> {
        phi.par_iter()
            .map(|&p| {
                let term1 = (((m * p / 4.0).cos() / a).abs()).powf(n2);
                let term2 = (((m * p / 4.0).sin() / b).abs()).powf(n3);
                (term1 + term2).powf(-1.0 / n1)
            })
            .collect()
    }

    /// Fast Hilbert Encoding (Iterative).
    #[pyo3(text_signature = "(x, y, order)")]
    pub fn hilbert_encode(&self, mut x: u32, mut y: u32, order: u32) -> u64 {
        let mut d = 0u64;
        let mut s = 1 << (order - 1);
        while s > 0 {
            let rx = (x & s > 0) as u32;
            let ry = (y & s > 0) as u32;
            d += (s as u64) * (s as u64) * ((3 * rx) ^ ry) as u64;
            self.rot(s, &mut x, &mut y, rx, ry);
            s /= 2;
        }
        d
    }

    /// Native HDC Holographic Projection.
    /// Parallelized over rows using Rayon.
    #[pyo3(signature = (params, shape, _order))]
    pub fn project_holographic_manifold(
        &self,
        py: Python<'_>,
        params: Vec<f32>,
        shape: (usize, usize),
        _order: u32,
    ) -> PyResult<Py<PyArray2<f32>>> {
        let (rows, cols) = shape;

        // Gielis Parameters
        let m = params[0];
        let a = params[1];
        let b = params[2];
        let n1 = params[3];
        let n2 = params[4];
        let n3 = params[5];

        const C1: u32 = 0x9E3779B1;
        const C2: u32 = 0x85EBCA6B;

        // Parallel Projection
        let data: Vec<f32> = (0..rows)
            .into_par_iter()
            .flat_map(|i| {
                let row_seed = (i as u32).wrapping_mul(C1);
                let mut row = Vec::with_capacity(cols);
                for j in 0..cols {
                    let z = row_seed ^ (j as u32).wrapping_mul(C2);
                    let phi = (z as f32 / 4294967295.0) * 2.0 * std::f32::consts::PI;
                    let t1 = (((m * phi / 4.0).cos() / a).abs()).powf(n2);
                    let t2 = (((m * phi / 4.0).sin() / b).abs()).powf(n3);
                    row.push((t1 + t2).powf(-1.0 / n1));
                }
                row
            })
            .collect();

        // Convert to Bound PyArray then unbind for Py<T>
        let array = PyArray1::from_vec_bound(py, data);
        let reshaped = array
            .reshape((rows, cols))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(reshaped.unbind())
    }
}

impl PyTMatrixEngine {
    /// Hilbert Rotation Utility.
    fn rot(&self, n: u32, x: &mut u32, y: &mut u32, rx: u32, ry: u32) {
        if ry == 0 {
            if rx == 1 {
                *x = n - 1 - *x;
                *y = n - 1 - *y;
            }
            std::mem::swap(x, y);
        }
    }
}
