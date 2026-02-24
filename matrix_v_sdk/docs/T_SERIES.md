# T-Series: Morphological DNA Engine

The T-Series engine implements **Morphological DNA** based on the Gielis Superformula. it generates matrices whose elements are determined by complex topographic and trigonometric manifolds.

## Theory of Operation

The T-Series views a matrix as a discretized coordinate space on a complex topographic surface.

### 1. Gielis Radius
Elements are resolved by calculating the radius of a super-shape at a specific angle determined by the $(i, j)$ coordinates.
$$r(\phi) = \left[ \sum_{k=2}^3 \left\| \frac{\text{trig}(\phi)}{a_k} \right\|^{n_k} \right]^{-1/n_1}$$

### 2. Space-Filling Encodings
The engine often pairs the Superformula with a **Hilbert Curve** encoding to map 2D matrix indices to 1D topographic "DNA" sequences.

## Usage

### Monolith Version
```python
from matrix_v_monolith import r_gielis, hilbert_encode

# Calculate radius for a specific angle
radius = r_gielis(phi=0.5, m=4, a=1, b=1, n1=1, n2=1, n3=1)

# Map 2D coordinate to 1D DNA index
idx = hilbert_encode(i=10, j=25, order=8)
```

### Standalone SDK
```python
from matrix_v_sdk.vl.substrate.tmatrix import TMatrixEngine

# Generate a topographical matrix (DNA sample)
dna_matrix = TMatrixEngine.morpheme(shape=(128, 128))
```

## Key Benefits
- **Topographical Synthesis**: Ideal for creating complex synthetic datasets or "procedural" weights.
- **Morphological Control**: Changing a few parameters ($m, n_1, n_2, \dots$) radically transforms the entire matrix structure.
- **Deterministic Complexity**: Allows for testing engines against highly non-linear but deterministic data.
