# Usage Guide — V-Series Matrix SDK

## Quick Start

```python
from vl.substrate.matrix import MatrixOmega, SymbolicDescriptor

omega = MatrixOmega()
```

`MatrixOmega` is the adaptive controller. It automatically selects the fastest engine for your workload.

---

## 1. Dense Numerical Multiplication

For standard floating-point matrices, pass list-of-lists. The controller routes to the Spectral or Block engine based on size.

```python
import numpy as np

A = np.random.rand(256, 256).tolist()
B = np.random.rand(256, 256).tolist()

result = omega.compute_product(A, B)
# Returns list-of-lists
```

**Complexity**: O(n²) via Spectral Projection for n > 100, O(n³) Block otherwise.

---

## 2. Symbolic / Exascale Matrices

For matrices too large to materialize (billions/trillions of dimensions), use `SymbolicDescriptor`:

```python
# Create trillion-scale descriptors (zero memory)
A = SymbolicDescriptor(rows=10**12, cols=10**12, signature=0xAAAA)
B = SymbolicDescriptor(rows=10**12, cols=10**12, signature=0xBBBB)

# O(1) symbolic multiplication
C = A.multiply(B)
print(C.rows, C.cols, hex(C.signature))

# JIT element access — resolve any single element in O(1)
val = C.resolve(999_999_999, 42)
print(f"C[999999999, 42] = {val:.6f}")
```

**Complexity**: O(1) for composition and element access.

---

## 3. XMatrix (Semantic HDC Engine)

For hyperdimensional computing and structural lineage tracking:

```python
from vl.substrate.matrix import XMatrix

x1 = XMatrix(100, 100, seed=1)
x2 = XMatrix(100, 100, seed=2)

# O(1) symbolic composition
x3 = x1.compose(x2)
print(f"Composed signature: {hex(x3.signature)}")

# Materialize for verification
matrix = x1.multiply_materialize(x2)
```

---

## 4. PrimeMatrix (Analytical Divisor)

For number-theoretic structure analysis:

```python
from vl.substrate.matrix import PrimeMatrix

p = PrimeMatrix(50, 50)

# O(1) element: P[i,j] = 1 if (i+1) divides (j+1)
val = p.get_element(2, 5)  # Does 3 divide 6? → 1
```

---

## 5. Holographic Consensus (Trinity)

For Byzantine-fault-tolerant verification of matrix results:

```python
from vl.substrate.vld_holographic import TrinityConsensus, Hypervector

trinity = TrinityConsensus(seed=0x777)

# Resolve truth from Law, Intention, and Event
result = trinity.resolve("Gravity", "Falling", 0xABC)
print(f"Truth: {result.label}")
print(f"Signature: {hex(result.signature())}")
```

---

## 6. Direct Strategy Selection

Override automatic selection by calling engines directly:

```python
from vl.substrate.v_matrix import VMatrix
from vl.substrate.acceleration import MMP_Engine

# Spectral (JL Projection) — O(n²)
spectral = VMatrix(mode="spectral")
result = spectral.matmul(A, B)

# RNS (Exact Arithmetic) — O(n³) but bit-exact
mmp = MMP_Engine()
result = mmp.multiply(A, B)
```

---

## 7. Rust Acceleration (Native)

After building with `maturin develop --release`:

```python
from vld_core import PyFeistelMemoizer, PySpectralProjector, PySymbolicDescriptor

# Native Feistel (~100x faster)
f = PyFeistelMemoizer(rounds=4)
seed = f.project_to_seed(0xDEADBEEF)

# Native Spectral Matmul (~30x faster)
sp = PySpectralProjector(target_dim=128, seed=42)
result = sp.matmul(A, B)

# Native Exascale Descriptor (~100x faster)
d = PySymbolicDescriptor(10**12, 10**12, 0x123)
val = d.resolve(0, 0)
block = d.materialize_block(0, 0, 10, 10)  # 10x10 window
```

---

## 8. T-Matrix (Morphological Substrate)

For extreme parameter compression and structural pruning:

```python
from vl.substrate.tmatrix import T_MatrixVLinear

# 170,000x Compressed Layer (Weight-Free)
layer = T_MatrixVLinear(in_features=1024, out_features=1024)

# Native Holographic Projection (Parallel Rust)
output = layer(input_tensor) 

# Resonant Shunting (90% Noise Reduction)
resonant_layer = T_MatrixVLinear(512, 512, mode='resonant')
output = resonant_layer(input_tensor)
```

**Complexity**: O(1) storage, O(n²) projection (O(n) if shunted).

---

## Engine Selection Summary

| Input Type | Engine | Complexity | When to Use |
| :--- | :--- | :--- | :--- |
| `list[list[float]]` small | Adaptive Block | O(n³) | n < 100 |
| `list[list[float]]` large | Spectral (VMatrix) | O(n²) | n > 100 |
| `SymbolicDescriptor` | Symbolic | O(1) | Infinite-scale descriptors |
| `XMatrix` | HDC Semantic | O(1) | Structural lineage |
| `PrimeMatrix` | Analytical | O(1) | Divisor analysis |
| `dict` with `"trinity"` | Holographic | O(1) | Byzantine verification |

---

## Running Benchmarks

```bash
# Industry comparison (vs NumPy)
python comprehensive_benchmarks.py

# Accuracy analysis
python accuracy_analysis.py

# Rust unit tests
cd rust_core
cargo test
```
