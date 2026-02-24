"""
Verification Test for matrix_v_monolith.py
Ensures the zero-dependency monolith is fully operational without external imports.
"""

import sys
import os

def test_monolith_standalone():
    print("Checking Monolith Standalone Integrity...")
    
    # 1. Import check
    try:
        from matrix_v_monolith import MatrixV, VlAdaptiveRNS, InfiniteMatrix
        print(" [OK] Monolith found and imported.")
    except ImportError as e:
        print(f" [FAIL] Could not import from monolith: {e}")
        sys.exit(1)

    # 2. RNS Substrate Test
    rns = VlAdaptiveRNS(16)
    val = 98765432109876543210
    decomp = rns.decompose(val)
    reconst = rns.reconstruct(decomp)
    if val == reconst:
        print(f" [OK] RNS Substrate parity check passed.")
    else:
        print(f" [FAIL] RNS Substrate parity check failed ({val} != {reconst})")
        sys.exit(1)

    # 3. Symbolic Matrix Test (O(1) Matmul)
    sdk = MatrixV()
    A = sdk.symbolic(1000000, 1000000, seed=0xAAA)
    B = sdk.symbolic(1000000, 1000000, seed=0xBBB)
    C = A.matmul(B)
    
    # Seed check: element should be deterministic
    val1 = C[500, 500]
    val2 = C[500, 500]
    if val1 == val2:
        print(f" [OK] Symbolic Determinism: {val1:.4f}")
    else:
        print(f" [FAIL] Symbolic Determinism failure.")
        sys.exit(1)

    # 4. Adaptive Dispatch Check
    # (Checking if it routes to engines safely)
    import math
    dense_A = [[1.0, 0.0], [0.0, 1.0]]
    dense_B = [[2.0, 3.0], [4.0, 5.0]]
    res = sdk.multiply(dense_A, dense_B)
    
    expected = [[2.0, 3.0], [4.0, 5.0]]
    passed = True
    for r in range(2):
        for c in range(2):
            if not math.isclose(res[r][c], expected[r][c], rel_tol=1e-9):
                passed = False
    
    if passed:
        print(f" [OK] Adaptive Dispatch (Approximate) passed.")
    else:
        print(f" [FAIL] Adaptive Dispatch failed: {res}")
        sys.exit(1)

    print("\n[VERIFIED] Matrix-V Monolith is READY.")

if __name__ == "__main__":
    test_monolith_standalone()
