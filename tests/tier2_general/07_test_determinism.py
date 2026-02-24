import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

def test_determinism():
    print("Tier 2: [07] Cross-Engine Determinism")
    omega = MatrixOmega()
    A = [[1.2, 3.4], [5.6, 7.8]]
    B = [[9.0, 0.1], [2.3, 4.5]]
    
    r1 = omega.compute_product(A, B)
    r2 = omega.compute_product(A, B)
    assert r1 == r2
    print("[PASS]")

if __name__ == "__main__":
    test_determinism()


