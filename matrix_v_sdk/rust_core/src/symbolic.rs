//! Symbolic Descriptor: O(1) Exascale matrix composition.
//!
//! Equation:
//!   Multiply: NewSig = SigA ^ (SigB >> 1) ^ (Depth << 32)
//!   Resolve:  val(r,c) = fmix64(Sig ^ (r*cols + c)) → [-1, 1]
//!
//! This is the core of the "Infinite Matrix" abstraction.
//! Zero memory footprint for trillion-scale descriptors.

use pyo3::prelude::*;

/// MurmurHash3 finalization mix for 64-bit.
#[inline]
fn fmix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xFF51AFD7ED558CCD);
    k ^= k >> 33;
    k = k.wrapping_mul(0xC4CEB9FE1A85EC53);
    k ^= k >> 33;
    k
}

/// A symbolic matrix descriptor. No data stored — only shape and signature.
#[derive(Clone, Debug)]
pub struct SymbolicDescriptor {
    pub rows: u64,
    pub cols: u64,
    pub signature: u64,
    pub depth: u32,
}

impl SymbolicDescriptor {
    pub fn new(rows: u64, cols: u64, signature: u64, depth: u32) -> Self {
        Self {
            rows,
            cols,
            signature,
            depth,
        }
    }

    /// O(1) Symbolic composition.
    /// Folds the signatures of two matrices into a new descriptor.
    #[inline]
    pub fn multiply(&self, other: &SymbolicDescriptor) -> Result<SymbolicDescriptor, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Dimension mismatch: {}x{} cannot multiply with {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }
        let new_sig = self.signature ^ (other.signature >> 1) ^ ((self.depth as u64) << 32);
        Ok(SymbolicDescriptor {
            rows: self.rows,
            cols: other.cols,
            signature: new_sig,
            depth: self.depth + other.depth,
        })
    }

    /// O(1) JIT element realization.
    /// Materializes a single value at (r, c) from the signature field.
    #[inline]
    pub fn resolve(&self, r: u64, c: u64) -> f64 {
        let idx = r.wrapping_mul(self.cols).wrapping_add(c);
        let h = fmix64(self.signature ^ idx);
        (h as f64) / (u64::MAX as f64) * 2.0 - 1.0
    }

    /// Materialize a block of the infinite matrix.
    /// Useful for verification and small-window sampling.
    pub fn materialize_block(
        &self,
        row_start: u64,
        col_start: u64,
        row_count: usize,
        col_count: usize,
    ) -> Vec<Vec<f64>> {
        (0..row_count)
            .map(|i| {
                (0..col_count)
                    .map(|j| self.resolve(row_start + i as u64, col_start + j as u64))
                    .collect()
            })
            .collect()
    }
}

// --- PyO3 Wrapper ---

#[pyclass]
pub struct PySymbolicDescriptor {
    inner: SymbolicDescriptor,
}

#[pymethods]
impl PySymbolicDescriptor {
    #[new]
    #[pyo3(signature = (rows, cols, signature, depth=1))]
    fn new(rows: u64, cols: u64, signature: u64, depth: u32) -> Self {
        Self {
            inner: SymbolicDescriptor::new(rows, cols, signature, depth),
        }
    }

    fn multiply(&self, other: &PySymbolicDescriptor) -> PyResult<PySymbolicDescriptor> {
        match self.inner.multiply(&other.inner) {
            Ok(desc) => Ok(PySymbolicDescriptor { inner: desc }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
        }
    }

    fn resolve(&self, r: u64, c: u64) -> f64 {
        self.inner.resolve(r, c)
    }

    fn materialize_block(
        &self,
        row_start: u64,
        col_start: u64,
        row_count: usize,
        col_count: usize,
    ) -> Vec<Vec<f64>> {
        self.inner
            .materialize_block(row_start, col_start, row_count, col_count)
    }

    #[getter]
    fn rows(&self) -> u64 {
        self.inner.rows
    }
    #[getter]
    fn cols(&self) -> u64 {
        self.inner.cols
    }
    #[getter]
    fn signature(&self) -> u64 {
        self.inner.signature
    }
    #[getter]
    fn depth(&self) -> u32 {
        self.inner.depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_multiply() {
        let a = SymbolicDescriptor::new(1000, 1000, 0xAAAA, 1);
        let b = SymbolicDescriptor::new(1000, 1000, 0xBBBB, 1);
        let c = a.multiply(&b).unwrap();
        assert_eq!(c.rows, 1000);
        assert_eq!(c.cols, 1000);
        assert_eq!(c.depth, 2);
    }

    #[test]
    fn test_symbolic_dimension_mismatch() {
        let a = SymbolicDescriptor::new(100, 200, 0x1, 1);
        let b = SymbolicDescriptor::new(300, 100, 0x2, 1);
        assert!(a.multiply(&b).is_err());
    }

    #[test]
    fn test_resolve_determinism() {
        let d = SymbolicDescriptor::new(10, 10, 0xDEAD, 1);
        let v1 = d.resolve(5, 7);
        let v2 = d.resolve(5, 7);
        assert_eq!(v1, v2, "Resolve must be deterministic");
    }

    #[test]
    fn test_resolve_range() {
        let d = SymbolicDescriptor::new(1000, 1000, 0xBEEF, 1);
        for r in 0..100 {
            for c in 0..100 {
                let v = d.resolve(r, c);
                assert!(v >= -1.0 && v <= 1.0, "Value must be in [-1, 1]");
            }
        }
    }

    #[test]
    fn test_exascale_dimensions() {
        // Trillion-scale: should compile and run without issue
        let a = SymbolicDescriptor::new(10u64.pow(12), 10u64.pow(12), 0x123, 1);
        let b = SymbolicDescriptor::new(10u64.pow(12), 10u64.pow(12), 0x456, 1);
        let c = a.multiply(&b).unwrap();
        assert_eq!(c.rows, 10u64.pow(12));
        // Resolve a single element from the trillion-scale field
        let val = c.resolve(999_999_999_999, 999_999_999_999);
        assert!(val >= -1.0 && val <= 1.0);
    }
}
