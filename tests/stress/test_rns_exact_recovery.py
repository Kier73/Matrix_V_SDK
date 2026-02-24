"""
Anchor-Navigate: RNS-Exact Recovery Proof
==========================================
Verifies that Anchor-Navigate achieves zero-error (arithmetic exactness)
for dense integer matrices when exact=True.

THEORY:
  For integer matrices A, B, the product C = A @ B has exact integer entries.
  Anchor-Navigate with RNS-Exact logic performs the CUR projection in 
  residue fields Z/pZ and reconstructs the result via CRT.
  
  Unlike float-based CUR, this should have ZERO error for full-rank matrices
  if the anchor size s meets the matrix rank (or dimension).

Run: python tests/stress/test_rns_exact_recovery.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from matrix_v_sdk.vl.substrate.anchor import AnchorNavigator

def test_exact_integer_recovery():
    print("\n" + "="*60)
    print("1. EXACT RECOVERY: Integer Dense (N=16, s=16)")
    print("="*60)
    
    N = 16
    S = 16
    np.random.seed(42)
    
    # Random integers in range [-10, 10]
    A = np.random.randint(-10, 11, size=(N, N)).astype(float)
    B = np.random.randint(-10, 11, size=(N, N)).astype(float)
    
    # Ground truth
    C_exact = A @ B
    
    # 1. Float-based Navigator (Standard)
    print("  [Float Path] Initializing...")
    nav_float = AnchorNavigator(A, B, anchor_size=S, exact=False)
    err_float = []
    for i in range(N):
        for j in range(N):
            err_float.append(abs(nav_float.navigate(i, j) - C_exact[i, j]))
    max_err_float = max(err_float)
    print(f"  Max Error (Float): {max_err_float:.2e}")
    
    # 2. RNS-Exact Navigator
    print("  [RNS-Exact Path] Initializing...")
    nav_exact = AnchorNavigator(A, B, anchor_size=S, exact=True, scale=1) # scale=1 for pure integers
    err_exact = []
    for i in range(N):
        for j in range(N):
            val = nav_exact.navigate(i, j)
            err_exact.append(abs(val - C_exact[i, j]))
            
    max_err_exact = max(err_exact)
    print(f"  Max Error (Exact): {max_err_exact:.2e}")
    
    if max_err_exact == 0.0:
        print("  [PASS] ZERO error achieved. Arithmetic exactness proven.")
    else:
        print(f"  [FAIL] Errors present. Exactness not achieved.")
        sys.exit(1)

def test_exact_fixed_point_recovery():
    print("\n" + "="*60)
    print("2. EXACT RECOVERY: Fixed-Point Dense (N=10, s=10, scale=100)")
    print("="*60)
    
    N = 10
    S = 10
    scale = 100
    np.random.seed(42)
    
    # Random floats with 2 decimal places
    A = np.round(np.random.uniform(-5, 5, size=(N, N)) * scale) / scale
    B = np.round(np.random.uniform(-5, 5, size=(N, N)) * scale) / scale
    
    C_exact = A @ B
    
    nav = AnchorNavigator(A, B, anchor_size=S, exact=True, scale=scale)
    
    errors = []
    for i in range(N):
        for j in range(N):
            errors.append(abs(nav.navigate(i, j) - C_exact[i, j]))
            
    max_err = max(errors)
    print(f"  Max Error (scale={scale}): {max_err:.2e}")
    
    if max_err < 1e-12: # Allowing for tiny float noise in C_exact (numpy) calculation
        print("  [PASS] Fixed-point exactness proven.")
    else:
        print("  [FAIL] Accuracy outside expected range.")
        sys.exit(1)

if __name__ == "__main__":
    test_exact_integer_recovery()
    test_exact_fixed_point_recovery()


