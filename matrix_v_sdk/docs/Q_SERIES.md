# Q-Series: Exascale Substrate Engine

The Q-Series (QMatrix) represents the **Substrate of Highest Complexity**. It uses quantum-inspired metrics, specifically **Von Neumann Entropy**, to quantify the information density of a matrix and route it accordingly.

## Theory of Operation

### 1. Quantum Rank Proxy
Standard rank computation is $O(N^3)$. The Q-Series uses S-entropy as a proxy to estimate rank/complexity in $O(N^2 \log N)$ time.
$$S = - \text{Tr}(\rho \log \rho)$$

### 2. Exascale Routing
When the dispatcher encounter matrices that are too large for standard engines but too significant for symbolic approximation, the QMatrix engine manages their lifecycle through tiered acceleration (often involving the Rust backend).

## Usage

### Monolith Version
```python
from matrix_v_monolith import QuantumRankProxy

# Estimate information density
ent = QuantumRankProxy.s_entropy(singular_values=[1.0, 0.5, 0.1])
```

### Standalone SDK
```python
from matrix_v_sdk.vl.substrate.matrix import QMatrix

# High-complexity exascale matrix
q = QMatrix(rows=8192, cols=8192)
```

## Key Benefits
- **Metric-Driven Dispatch**: Entropy-based routing ensures the most complex data gets the most robust acceleration.
- **Scalability**: Designed to handle the "gap" between low-rank manifolds and dense spectral substrates.
- **Information Parity**: Ensures that critical information (rank) is preserved during compression or sharding.
