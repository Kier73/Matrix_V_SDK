# MatrixOmega: Adaptive Dispatch Logic

`MatrixOmega` is the central "brain" of the Matrix-V SDK. It acts as an intelligent router, analyzing input matrices and selecting the optimal mathematical engine based on structural feature vectors.

## Dispatch Strategy

The dispatcher follows a hierarchical decision tree to minimize computational complexity while maintaining maximum parity.

### 1. Feature Extraction
Before computation, the system samples $O(1)$ elements to compute a `MatrixFeatureVector`:
- **Sparsity**: Measured via zero-count sampling.
- **Periodicity**: Detects repeating tile patterns (cyclical redundancy).
- **Row Variance**: Identifies low-rank manifolds where energy is concentrated in a few components.

### 2. Decision Logic

The dispatcher routes based on the following prioritizations:

| Feature Condition | Selected Engine | Benefit |
| :--- | :--- | :--- |
| **Periodicity > 0.5** | G-Series (Inductive) | $O(1)$ cache hits for redundant tiles. |
| **Sparsity > 0.7** | V-Series (Spectral) | $O(N^2 \log N)$ Matmul in subspace. |
| **Variance < 0.1** | A-Series (Anchor) | $O(s^2 N)$ realization via CUR. |
| **Default** | MMP / Standard | High-stability modular arithmetic fallback. |

## Usage

```python
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

omega = MatrixOmega()

# The dispatcher handles analysis and routing automatically
result = omega.compute_product(A, B)
```

## Internal Workflow
1. **Sample**: Check structural features of $A$ and $B$.
2. **Strategy Cache**: Consult history to see if similar shapes/features were successfully handled.
3. **Execute**: Dispatch to the chosen engine $(\mathcal{E}_A, \mathcal{E}_V, \mathcal{E}_G, \dots)$.
4. **Learn**: Record execution time and accuracy to refine future decisions.
