# Codebase Map: Matrix-V SDK

This document provides a high-level structural map of the Matrix-V SDK repository.

## Root Directory
- `matrix_v_monolith.py`: Zero-dependency, standalone version of the SDK.
- `pyproject.toml`: Modern packaging configuration.
- `README.md`: Entry point for hardware/software requirements and quick start.
- `LICENSE`: MIT License.

## Package: `matrix_v_sdk/`
The core namespace for the SDK.

### `vl/` (Virtual Layer Substrate)
- `substrate/rns_signature.py`: Residue Number System (RNS) arithmetic and algebraic signatures.
- `substrate/matrix.py`: Unified `MatrixOmega` dispatcher and `InfiniteMatrix` symbolic descriptors.
- `substrate/anchor.py`: A-Series Geometric Navigation (CUR Decomposition).
- `substrate/acceleration.py`: V-Series Spectral Projector (Johnson-Lindenstrauss).
- `math/primitives.py`: Feistel ciphers and MurmurHash3 fmix64 implementations.

### `extensions/` (Framework Bridges)
- `torch_bridge.py`: PyTorch `nn.Module` and Autograd integration.
- `jax_bridge.py`: JAX functional wrappers and custom VJPs.
- `numba_bridge.py`: Numba JIT acceleration for P-Series lattices.
- `hf_bridge.py`: HuggingFace Transformer/Attention acceleration.

### `rust_core/` (High-Performance Backend)
- Rust source code and `Cargo.toml` for the compiled acceleration backend.

### `docs/`
- `architecture.md`: Low-level design of the RNS and manifolds.
- `USAGE.md`: Detailed API documentation for each series.
- `INSTALL.md`: Setup guide for various platforms.

## Development & Test
- `tests/`: Multi-tiered testing suite (Tier 1 Basic -> Tier 4 Impossible).
- `examples/`: Reference implementations for training and benchmarking.
- `benchmarks/`: Systematic scalability and accuracy comparisons.
