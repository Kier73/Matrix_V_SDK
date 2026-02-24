# SDK Academic Experiments: The O(N^2) Probe

This directory contains rigorous experimental scripts designed to provide academic-level proof or disproof for the sub-cubic complexity claims of the Matrix-V SDK.

## Experimental Framework
Each experiment focuses on a specific mathematical "Collapse" engine and measures:
1. **Time Complexity ($T$ vs $N$)**: Regression analysis to determine the empirical exponent.
2. **Precision Stability ($\epsilon$ vs $N$)**: Relationship between scale and error-rate.
3. **Manifold Boundary**: The point at which the acceleration strategy becomes more efficient than the "N^3 Wall."

## The Experiments
- **Exp 01: Johnson-Lindenstrauss Limit**: Rigorous test of $O(N^2 \log N)$ spectral projection.
- **Exp 02: Symbolic Field Convergence**: Proof-of-concept for $O(1)$ symbolic matmul synthesis.
- **Exp 03: Number-Theoretic Manifolds**: Verification of $O(N^2)$ resolution for Dirichlet-structured matrices.
- **Exp 04: The Master Solver**: A unified engine integration test that automatically selects the lowest-complexity manifold path.

---
**Goal**: To build a verifiable dataset that supports a high-level whitepaper on "Resonant Matrix Solvers."
