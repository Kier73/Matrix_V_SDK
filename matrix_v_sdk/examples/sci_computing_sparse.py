"""
Example: SciPy Sparse Bridge (Industrial FEM Logic)
===================================================

This example demonstrates how the Matrix SDK can be used in scientific 
computing contexts, specifically for finite element method (FEM) problems 
where matrices are initially sparse (stiffness matrices) but might 
undergo dense operations during solving or preconditioning.

Key SDK Integration:
  - scipy_bridge.py: Converts CSR/CSC to the Virtual Layer
  - O(1) Sparsity Extraction: Bypasses dense scanning
  - Adaptive Routing: Switches engine based on mesh density
"""
import sys
import os
import numpy as np

# Ensure the parent of the SDK root is on the path so 'import matrix_v_sdk' works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import scipy.sparse as sp
    from matrix_v_sdk.extensions.scipy_bridge import SparseMatrixV
except ImportError:
    print("Error: scipy is required for this example. Run 'pip install scipy'")
    sys.exit(1)

def run_fem_simulation_logic():
    print("--- 1. Initializing Sparse Stiffness Matrix (N=1000) ---")
    # Simulate a sparse stiffness matrix for a 1D/2D mesh
    n = 1000
    density = 0.005  # 0.5% fill rate
    K = sp.random(n, n, density=density, format='csr')
    
    # Simulate a vector or small dense block of forces/displacement
    # (e.g., during a sub-domain integration)
    F = sp.random(n, 10, density=0.1, format='csr')
    
    smv = SparseMatrixV()
    
    # 1. Feature Extraction (O(1))
    print("\nExecuting Feature Extraction...")
    fv = smv.feature_vector(K, F)
    print(f"  Matrix Class: {fv.shape_class()}")
    print(f"  Sparsity Detected: {fv.sparsity:.4f} (nnz={K.nnz})")
    print(f"  Flop Cost Estimate: {fv.flop_cost:,}")

    # 2. Multiplication via Matrix SDK
    print("\nSolving Local Force System (Stiffness * Displacement)...")
    # In a real FEM, you'd solve Ax=b, but here we show the accelerated 
    # dense-sparse accumulation logic through the SDK
    result_csr = smv.multiply(K, F, output_format='csr')
    
    print(f"  Result Type: {type(result_csr)}")
    print(f"  Result NNZ: {result_csr.nnz}")
    
    # 3. Dense region detection
    # If the mesh gets refined locally, density increases
    print("\n--- 2. High Density mesh Refinement (Density -> 40%) ---")
    K_refined = sp.random(n, n, density=0.4, format='csr')
    fv_refined = smv.feature_vector(K_refined, F)
    
    # Adaptive routing will pick a different engine for dense vs sparse
    # (Spectral/Sparse-optimised vs Block-optimised)
    print(f"  Refined Sparsity: {fv_refined.sparsity:.4f}")
    result_refined = smv.multiply(K_refined, F, output_format='dense')
    print(f"  Refined Result Max Value: {np.max(result_refined):.4f}")

if __name__ == "__main__":
    run_fem_simulation_logic()


