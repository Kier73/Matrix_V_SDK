//! VLD Core: Native Rust acceleration for the V-Series Matrix SDK.
//!
//! This crate provides high-performance implementations of:
//! - Feistel Cipher (Memoization & Grounding)
//! - RNS Engine (Chinese Remainder Theorem)
//! - Hypervector Logic (4096-bit HDC)
//! - Spectral Projection (JL-Lemma)
//! - Symbolic Descriptors (O(1) Exascale)

pub mod feistel;
pub mod hypervector;
pub mod rns;
pub mod spectral;
pub mod symbolic;
pub mod tmatrix;

use pyo3::prelude::*;

/// The Python module exposed via PyO3.
#[pymodule]
fn vld_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<feistel::PyFeistelMemoizer>()?;
    m.add_class::<rns::PyRNSEngine>()?;
    m.add_class::<hypervector::PyHypervector>()?;
    m.add_class::<spectral::PySpectralProjector>()?;
    m.add_class::<symbolic::PySymbolicDescriptor>()?;
    m.add_class::<tmatrix::PyTMatrixEngine>()?;
    Ok(())
}
