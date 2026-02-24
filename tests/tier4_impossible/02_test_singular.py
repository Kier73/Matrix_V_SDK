import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

def test_singular():
    print("Tier 4: [02] Extreme Singular Condition Numbers")
    omega = MatrixOmega()
    # Matrix with a single non-zero row (Rank 1)
    A = [[0.0]*100 for _ in range(100)]
    A[0] = [1.0]*100
    B = [[1.0]*100 for _ in range(100)]
    
    res = omega.compute_product(A, B)
    # Expected: Row 0 is all 100.0, rest 0.0
    assert sum(res[0]) == 10000.0
    assert sum(res[1]) == 0.0
    print("[PASS]")

if __name__ == "__main__":
    test_singular()


