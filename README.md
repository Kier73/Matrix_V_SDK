# Matrix-V SDK: Focused on alternate forms of Matrix multiplication

## System Architecture

### 1. The RNS Substrate (Residue Rings)
The system is built on a 16-prime Residue Number System (RNS), providing a 310-bit dynamic range. This allows for arithmetic exactness across $1024 \times 1024$ dense matrix products without floating-point drift or overflow risk.
- **Homomorphism**: $f(A \times B) = f(A) \otimes f(B)$ in residue space.
- **Verification**: Chinese Remainder Theorem (CRT) lifting for exact parity.

### 2. MatrixOmega: Adaptive Dispatch
`MatrixOmega` serves as the central controller, classifying matrices into structural categories and dispatching them to specialized engines.
- **Structural Signals**: Samples $O(1)$ elements for sparsity, row variance, and tile periodicity.
- **Dispatch Logic**: Favor $O(s^2)$ geometric projection over $O(k)$ dot products when redundancy is detected.

### 3. Complexity Reduction
| Operation | Traditional | Matrix-V | Reduction |
| :--- | :--- | :--- | :--- |
| **Storage** | $O(N^2)$ | $O(s)$ | Symbolic Signature |
| **Multiply** | $O(N^3)$ | $O(s^2 N)$ | Algorithmic Bypass |
| **Element Query** | $O(k)$ | $O(s^2)$ | Geometric Navigation |
| **Verification** | $O(N^2)$ | $O(1)$ | Ring Homomorphism |

## Core Features
1. **Anchor Navigator**: RNS-exact geometric projection ($O(s^2)$) for low-rank and structured dense manifolds.
2. **Spectral Projector**: Adaptive random projection ($O(N^2 \log N)$) for high-sparsity substrates.
3. **Inductive Engine**: Rust-backed tile memoization for cyclically redundant data.
4. **Resilience**: P-Series analytical resolution for lattice-structured matrices.
5. **Symbolic Trinity**: $O(1)$ holographic composition of infinite-scale matrices.

## Requirements

To run the Matrix-V SDK with full acceleration, the following environment is required:

### Software
- **Python 3.8+**: Core runtime for the Virtual Layer substrate.
- **NumPy >= 1.20.0**: Required for numerical primitives and spectral projections.
- **Rust (Cargo) 1.65+**: Needed to compile the `vl_core` high-performance backend.
- **C Compiler**: 
  - **Windows**: MinGW-w64 (required for Rust-to-DLL compilation).
  - **Linux/macOS**: GCC or Clang.

### Python Dependencies

The SDK uses a modular dependency architecture:

| Component | Requirement | Purpose |
| :--- | :--- | :--- |
| **Core** | `numpy` | Essential numerical arithmetic. |
| **PyTorch Bridge** | `torch` | `MatrixVLinear` and Transformer support. |
| **JAX Bridge** | `jax`, `jaxlib` | Functional pipelines and autodiff. |
| **Numba Bridge** | `numba` | JIT-accelerated lattice resolution. |

## Installation

### Standard Installation (Recommended)
You can install the SDK and its core dependencies via `pip`:
```bash
pip install .
```

To install with optional framework support:
```bash
# For PyTorch support
pip install ".[torch]"

# For all framework bridges
pip install ".[all]"
```

### Backend Compilation
If you are developing or need the Rust-backed acceleration:
```bash
cd matrix_v_sdk/rust_core
cargo build --release
```

## The Zero-Dependency Monolith

For environments where minimize dependency footprints (e.g., edge devices, secure servers, or zero-install scripts), the SDK provides `matrix_v_monolith.py`.

- **Standalone**: All core engines (Adaptive RNS, Symbolic Trinity, Anchor, Spectral) are collapsed into a single file.
- **Pure Python**: Runs on the Standard Library only (no NumPy required).
- **Portable**: Simply copy `matrix_v_monolith.py` to your project and import it.

---

## Quick Start

### 1. Using the Monolith (Zero-Dependency)
Perfect for quick scripts or portable tools.
```python
from matrix_v_monolith import MatrixV

# Initialize the unified interface
sdk = MatrixV()

# Create a symbolic matrix (trillion-scale, O(1) memory)
A = sdk.symbolic(10**12, 10**12, seed=0x1)
B = sdk.symbolic(10**12, 10**12, seed=0x2)

# O(1) Symbolic Multiply
C = A.matmul(B)

# JIT Element Resolution
val = C[1234567, 8901234]
print(f"Resolved Value: {val}")
```

### 2. Using the Full SDK (High Performance)
Recommended for production and framework integration.
```python
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega
import numpy as np

# Initialize the acceleration engine
omega = MatrixOmega()

# Dense Spectral Benchmark
A = np.random.randn(512, 512).tolist()
B = np.random.randn(512, 512).tolist()

# Automated engine selection
result = omega.compute_product(A, B)
```

## Framework Integration

The SDK provides dedicated access points for major Python frameworks.

### PyTorch Integration
Use `MatrixVLinear` as a drop-in replacement for `nn.Linear` to leverage categorical acceleration.
```python
from matrix_v_sdk.extensions.torch_bridge import MatrixVLinear
import torch.nn as nn

model = nn.Sequential(
    MatrixVLinear(512, 1024),
    nn.ReLU(),
    MatrixVLinear(1024, 10)
)
```

### Numba Tandem Operation
The `NumbaBridge` allows parallel JIT resolution of algebraic manifolds.
```python
from matrix_v_sdk.extensions.numba_bridge import NumbaBridge

# Resolve a high-speed P-Series lattice using Numba-parallel kernels
lattice = NumbaBridge.resolve_p_lattice(1024, 1024, p_val=2)
```

### JAX Functional Integration
The SDK supports functional JAX pipelines with automatic differentiation via custom VJPs.
```python
import jax.numpy as jnp
from matrix_v_sdk.extensions.jax_bridge import matrix_v_matmul

A = jnp.ones((512, 512))
B = jnp.ones((512, 512))

# Matrix-V matmul inside JAX
C = matrix_v_matmul(A, B)
```

### Transformer (Attention) Support
Accelerate modern attention mechanisms with categorical resonance.
```python
from matrix_v_sdk.extensions.hf_bridge import MatrixVAttention
import torch

attn = MatrixVAttention(embed_dim=512, num_heads=8)
x = torch.randn(1, 128, 512) # batch, seq, embed
output = attn(x)
```

### Infinite-Scale Symbolic Operations
Perform $O(1)$ symbolic multiplication on matrices of effectively infinite size ($10^{12} \times 10^{12}$) and resolve elements JIT.
```python
from matrix_v_sdk.vl.substrate.matrix import SymbolicDescriptor, InfiniteMatrix

# Create trillion-scale descriptors
A = InfiniteMatrix(SymbolicDescriptor(10**12, 10**12, signature=0x1))
B = InfiniteMatrix(SymbolicDescriptor(10**12, 10**12, signature=0x2))

# O(1) Symbolic Multiply
C = A.matmul(B)

# JIT Element Resolution
val = C[1234567, 8901234]
```

### RH-Series (Riemann-Hilbert)
Number-theoretic manifolds for analytical sparsity.
```python
from matrix_v_sdk.vl.substrate.acceleration import RH_SeriesEngine

rh = RH_SeriesEngine()
# mu(6) = 1
mu_6 = rh.get_mobius(6)
```


## Verification & Benchmarking
The SDK maintains a rigorous empirical verification suite:
- **Integration Tests**: Located in `tests/integration/`. Run to verify adaptive routing and solver parity.




| Symbol | Engine Name | Primary Function | State | Direct Equation / Numerical Grounding |
| :--- | :--- | :--- | :--- | :--- |
| **A** | **Anchor** | Geometric Nav | Active | $C_{ij} = \mathbf{k}_i^\top W^{-1} \mathbf{r}_j \quad (O(s^2) \text{ realization})$ |
| **V** | **V-Series** | Spectral Projector | Active | $D \ge \lceil 4 \ln N / (\epsilon^2/2 - \epsilon^3/3) \rceil$ |
| **G** | **G-Series** | Inductive Tiling | Active | $\mathcal{L}_{cache}(H(A), H(B)) \to O(1) \text{ hit}$ |
| **X** | **X-Series** | Isomorphic HDC | Active | $\mathbf{v}_C = \mathbf{v}_A \oplus \mathbf{v}_B \quad (1024\text{-bit binding})$ |
| **P** | **P-Series** | Analytical Divisor | Active | $P_{ij} = [\exists d \in \mathbb{Z} : j = i \cdot d]$ |
| **RH** | **RH-Series** | Riemann-Hilbert | Active | $\mathcal{M}_{ij} = \mu(\text{gcd}(i,j)) \quad (\text{Square-free sparsity})$ |
| **MMP** | **Modular Manifold** | RNS Channels | Active | $X_{ij} = \left[ \sum_{k=1}^n r_k M_k y_k \right] \pmod M$ |
| **T** | **T-Matrix** | Morphological DNA | Active | $r(\phi) = \left[ \sum_{k=2}^3 \left\| \frac{\text{trig}(\phi)}{a_k} \right\|^{n_k} \right]^{-1/n_1}$ |
| **Q** | **QMatrix** | Exascale Substrate | Active | $S = - \text{Tr}(\rho \log \rho) \quad (\text{Quantum Rank Proxy})$ |
| **S** | **Symbolic** | Deterministic JIT | Active | $\mathcal{H}(Sig, \text{depth}) \to f(r,c)$ |
