# Installation Guide — V-Series Matrix SDK

## Prerequisites

| Tool | Version | Purpose |
| :--- | :--- | :--- |
| Python | 3.10+ | Core SDK runtime |
| pip | latest | Package management |
| Rust | 1.75+ | Native acceleration (optional) |
| Cargo | 1.75+ | Rust build system (optional) |

---

## 1. Python SDK (Core)

### Clone & Install Dependencies

```bash
cd matrix_v_sdk
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "from vl.substrate.matrix import MatrixOmega; print('SDK Ready')"
```

Expected output:
```
[GVM Engine] Gen 8 Backend Online (order=10, seed=0x2a)
[GVM Engine] Hydrating 2D Cache (1024x1024)...
[GVM Engine] Cache Ready (Vectorized Path Active).
SDK Ready
```

---

## 2. Rust Acceleration Layer (Optional, Recommended)

The Rust layer provides 10x–100x speedups for cryptographic, symbolic, and spectral operations.

### Install Rust Toolchain

```bash
# Windows (via rustup)
winget install Rustlang.Rustup

# Linux/macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build the Native Extension

```bash
cd rust_core
pip install maturin
maturin develop --release
```

### Verify Native Module

```python
from vld_core import PyFeistelMemoizer, PySymbolicDescriptor, PyTMatrixEngine

f = PyFeistelMemoizer()
seed = f.project_to_seed(0xDEADBEEF)
print(f"Feistel Seed: {hex(seed)}")

d = PySymbolicDescriptor(10**12, 10**12, 0x123)
val = d.resolve(999_999_999, 0)
print(f"Exascale Element: {val:.6f}")

t = PyTMatrixEngine()
manifold = t.project_holographic_manifold([4.0, 1.0, 1.0, 0.5, 0.5, 0.5], (128, 128), 7)
print(f"Native T-Matrix Manifold: {manifold.shape}")
```

### Run Rust Unit Tests

```bash
cd rust_core

# Windows PowerShell
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY="1"; cargo test

# Linux/macOS
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test
```

Expected: `test result: ok. 13 passed; 0 failed`

---

## 3. Optional Dependencies

| Package | Install | Used By |
| :--- | :--- | :--- |
| `torch` | `pip install torch` | KinematicEngine (4D projections) |
| `scipy` | `pip install scipy` | Sparse matrix benchmarks |
| `matplotlib` | `pip install matplotlib` | Visualization & plots |

---

## Troubleshooting

| Issue | Fix |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'vl'` | Run from the `matrix_v_sdk/` root directory |
| `PyO3: Python version too new` | Set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` |
| `cargo: command not found` | Install Rust via [rustup.rs](https://rustup.rs) |
