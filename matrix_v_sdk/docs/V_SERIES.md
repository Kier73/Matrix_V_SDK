# V-Series: Spectral Projector

The V-Series engine implements **Adaptive Random Projection** based on the Johnson-Lindenstrauss (JL) Lemma. It is designed for high-sparsity substrates where traditional methods are inefficient.

## Theory of Operation

Spectral Projection maps high-dimensional matrix products into a lower-dimensional subspace while preserving the geometry of the manifold.

### 1. Random Projection Matrix
The engine generates a deterministic projection matrix $\Omega$ using the Feistel-Murmur substrate. This ensures consistency across different nodes in a distributed system.

### 2. Dimension Reduction
Given a target dimension $d \ll N$, matrices are projected:
$$A' = A \times \Omega, \quad B' = \Omega^\top \times B$$

### 3. Matmul in Subspace
The product is computed in the compressed space:
$$C \approx A' \times B'$$

## Usage Example

```python
from matrix_v_sdk.vl.substrate.acceleration import SpectralProjector

# Initialize the projector
projector = SpectralProjector(target_dim=128)

# Compute product via subspace matmul
result = projector.matmul(A, B)
```

## Key Benefits
- **Complexity**: $O(N^2 \log N)$ or less depending on sparsity.
- **Privacy Preserving**: Projections act as a natural hashing layer, obscuring raw weights.
- **Scalability**: Excellent performance for very large, very sparse matrices that don't fit in standard caches.
