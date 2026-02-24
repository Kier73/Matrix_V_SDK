//! RNS Engine: Residue Number System for parallel, bit-exact arithmetic.
//!
//! Equation: x = {r1, r2, ..., rn} mod {m1, m2, ..., mn}
//! CRT Reconstruction: x = sum(ri * Mi * yi) mod M
//!
//! Uses native u128 to avoid BigInt overhead entirely.

use pyo3::prelude::*;

/// Standard prime moduli for 64-bit Dynamic Range.
const MODULI: [u64; 8] = [251, 257, 263, 269, 271, 277, 281, 283];
const INVERSES: [u64; 8] = [224, 59, 81, 129, 259, 218, 198, 182];

/// Product of all moduli. Precomputed.
const DYNAMIC_RANGE: u128 = 27_243_110_295_742_882_889;

pub struct RNSEngine;

impl RNSEngine {
    /// Decompose a value into its residues modulo each prime.
    #[inline]
    pub fn to_residues(val: u128) -> [u64; 8] {
        let mut res = [0u64; 8];
        for (i, &m) in MODULI.iter().enumerate() {
            res[i] = (val % m as u128) as u64;
        }
        res
    }

    /// Reconstruct from residues via CRT.
    #[inline]
    pub fn from_residues(residues: &[u64; 8]) -> u128 {
        let mut result: u128 = 0;
        for i in 0..8 {
            let mi = MODULI[i] as u128;
            let big_m_i = DYNAMIC_RANGE / mi;
            let inv = INVERSES[i] as u128;
            let term = ((residues[i] as u128) * big_m_i % DYNAMIC_RANGE * inv) % DYNAMIC_RANGE;
            result = (result + term) % DYNAMIC_RANGE;
        }
        result
    }

    /// Parallel element-wise multiply in RNS domain.
    /// This is O(1) per element since we only operate on small residues.
    #[inline]
    pub fn multiply_residues(a: &[u64; 8], b: &[u64; 8]) -> [u64; 8] {
        let mut res = [0u64; 8];
        for i in 0..8 {
            res[i] = (a[i] as u128 * b[i] as u128 % MODULI[i] as u128) as u64;
        }
        res
    }

    /// Add in RNS domain.
    #[inline]
    pub fn add_residues(a: &[u64; 8], b: &[u64; 8]) -> [u64; 8] {
        let mut res = [0u64; 8];
        for i in 0..8 {
            res[i] = ((a[i] as u64 + b[i] as u64) % MODULI[i] as u64) as u64;
        }
        res
    }
}

// --- PyO3 Wrapper ---

#[pyclass]
pub struct PyRNSEngine;

#[pymethods]
impl PyRNSEngine {
    #[new]
    fn new() -> Self {
        Self
    }

    fn to_residues(&self, val: u128) -> Vec<u64> {
        RNSEngine::to_residues(val).to_vec()
    }

    fn from_residues(&self, residues: Vec<u64>) -> u128 {
        let mut arr = [0u64; 8];
        for (i, &v) in residues.iter().enumerate().take(8) {
            arr[i] = v;
        }
        RNSEngine::from_residues(&arr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rns_roundtrip() {
        let val: u128 = 123_456_789;
        let residues = RNSEngine::to_residues(val);
        let recovered = RNSEngine::from_residues(&residues);
        assert_eq!(val, recovered, "RNS roundtrip must be lossless");
    }

    #[test]
    fn test_rns_multiply() {
        let a: u128 = 100;
        let b: u128 = 200;
        let ra = RNSEngine::to_residues(a);
        let rb = RNSEngine::to_residues(b);
        let rc = RNSEngine::multiply_residues(&ra, &rb);
        let result = RNSEngine::from_residues(&rc);
        assert_eq!(result, a * b, "RNS multiply must equal direct multiply");
    }
}
