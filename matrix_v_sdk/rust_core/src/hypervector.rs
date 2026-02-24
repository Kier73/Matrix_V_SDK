//! Hypervector: 4096-bit High-Dimensional Computing logic.
//!
//! Equation: Bind(A, B) = A XOR B
//!           Bundle(v1..vn) = Majority(v1..vn)
//!
//! Uses [u64; 64] representation (64 * 64 = 4096 bits).
//! Majority voting is parallelized via Rayon.

use pyo3::prelude::*;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

const CHUNKS: usize = 64; // 64 x 64-bit = 4096 bits

/// A 4096-bit hypervector stored as 64 u64 chunks.
#[derive(Clone)]
pub struct Hypervector {
    pub data: [u64; CHUNKS],
    pub label: String,
}

impl Hypervector {
    /// Generate a deterministic hypervector from a seed.
    pub fn from_seed(seed: u64, label: &str) -> Self {
        let mut data = [0u64; CHUNKS];
        let seed_bytes = seed.to_le_bytes();

        for i in 0..CHUNKS {
            let mut hasher = Sha256::new();
            hasher.update(&seed_bytes);
            hasher.update(&(i as u64).to_le_bytes());
            let hash = hasher.finalize();

            // XOR-fold 256-bit hash → 64-bit chunk
            let a = u64::from_le_bytes(hash[0..8].try_into().unwrap());
            let b = u64::from_le_bytes(hash[8..16].try_into().unwrap());
            let c = u64::from_le_bytes(hash[16..24].try_into().unwrap());
            let d = u64::from_le_bytes(hash[24..32].try_into().unwrap());
            data[i] = a ^ b ^ c ^ d;
        }

        Self {
            data,
            label: label.to_string(),
        }
    }

    /// XOR Binding: O(64) = O(1) relative to dimension.
    #[inline]
    pub fn bind(&self, other: &Hypervector) -> Hypervector {
        let mut data = [0u64; CHUNKS];
        for i in 0..CHUNKS {
            data[i] = self.data[i] ^ other.data[i];
        }
        Hypervector {
            data,
            label: format!("({} ⊕ {})", self.label, other.label),
        }
    }

    /// Bitwise Majority Vote: Byzantine consensus.
    /// For each bit position, the value with >50% votes wins.
    /// Parallelized across chunks via Rayon.
    pub fn majority_bundle(vectors: &[Hypervector]) -> Hypervector {
        let n = vectors.len();
        if n == 0 {
            return Hypervector {
                data: [0; CHUNKS],
                label: "Empty".into(),
            };
        }
        if n == 1 {
            return vectors[0].clone();
        }

        let threshold = n / 2;

        // Parallel across all 64 chunks
        let data: Vec<u64> = (0..CHUNKS)
            .into_par_iter()
            .map(|chunk_idx| {
                let mut votes = [0u32; 64];
                for v in vectors {
                    let chunk = v.data[chunk_idx];
                    for bit in 0..64 {
                        if (chunk >> bit) & 1 == 1 {
                            votes[bit] += 1;
                        }
                    }
                }
                let mut result: u64 = 0;
                for bit in 0..64 {
                    if votes[bit] as usize > threshold {
                        result |= 1 << bit;
                    }
                }
                result
            })
            .collect();

        let mut result_data = [0u64; CHUNKS];
        result_data.copy_from_slice(&data);

        Hypervector {
            data: result_data,
            label: format!("Majority({} nodes)", n),
        }
    }

    /// FNV-1a style signature collapse: 4096 → 64 bits.
    pub fn signature(&self) -> u64 {
        let mut h: u64 = 0xCBF29CE484222325;
        for &chunk in &self.data {
            h ^= chunk;
            h = h.wrapping_mul(0x100000001B3);
        }
        h
    }

    /// Hamming distance between two hypervectors.
    pub fn hamming_distance(&self, other: &Hypervector) -> u32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

// --- PyO3 Wrapper ---

#[pyclass]
pub struct PyHypervector {
    inner: Hypervector,
}

#[pymethods]
impl PyHypervector {
    #[new]
    fn new(seed: u64, label: &str) -> Self {
        Self {
            inner: Hypervector::from_seed(seed, label),
        }
    }

    fn bind(&self, other: &PyHypervector) -> PyHypervector {
        PyHypervector {
            inner: self.inner.bind(&other.inner),
        }
    }

    fn signature(&self) -> u64 {
        self.inner.signature()
    }

    fn hamming_distance(&self, other: &PyHypervector) -> u32 {
        self.inner.hamming_distance(&other.inner)
    }

    #[getter]
    fn label(&self) -> &str {
        &self.inner.label
    }

    #[staticmethod]
    fn majority_bundle(vectors: Vec<PyRef<PyHypervector>>) -> PyHypervector {
        let hvs: Vec<Hypervector> = vectors.iter().map(|v| v.inner.clone()).collect();
        PyHypervector {
            inner: Hypervector::majority_bundle(&hvs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bind_self_inverse() {
        let hv = Hypervector::from_seed(42, "Test");
        let bound = hv.bind(&hv);
        // XOR with self = 0
        assert!(bound.data.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_majority_odd() {
        let a = Hypervector::from_seed(1, "A");
        let b = Hypervector::from_seed(1, "B"); // same as A
        let c = Hypervector::from_seed(2, "C"); // different
        let result = Hypervector::majority_bundle(&[a.clone(), b, c]);
        // Majority should be closer to A
        assert_eq!(result.signature(), a.signature());
    }
}
