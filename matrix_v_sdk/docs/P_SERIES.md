# P-Series: Analytical Divisor Engine

The P-Series engine exploits **Lattice Divisibility** to resolve matrix elements analytically. It is specifically designed for matrices where the value of an element $(i, j)$ depends purely on their number-theoretic relationship.

## Theory of Operation

The engine bypasses $O(N^3)$ matmul entirely by resolving elements at the point of access.

### 1. Divisor Resolution
An element at index $(i, j)$ is resolved as a boolean or weighted divisor check:
$$P_{ij} = [j \equiv 0 \pmod{i}]$$

### 2. Lattice Geometry
This creates a structured "staircase" or lattice manifold. The engine provides specialized kernels (Numba-accelerated in the SDK) to resolve these lattices for use in spectral analysis or masking.

## Usage

### Monolith Version
```python
from matrix_v_monolith import PSeriesEngine

val = PSeriesEngine.resolve_divisor(i=2, j=4) # Returns 1
val = PSeriesEngine.resolve_divisor(i=3, j=4) # Returns 0
```

### Standalone SDK
```python
from matrix_v_sdk.vl.substrate.prime_matrix import PSeriesEngine

# Generate a high-speed divisor lattice
lattice = PSeriesEngine.generate_lattice(rows=1024, cols=1024)
```

## Key Benefits
- **O(1) Resolution**: The computational cost of an element does not scale with matrix size.
- **Structural Sparsity**: Ideal for generating masks or filters for number-theoretic signal processing.
- **Analytic Parity**: No rounding error; the divisibility check is always exact.
