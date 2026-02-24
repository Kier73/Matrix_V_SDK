import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

def test_errors():
    print("Tier 2: [06] Robustness Against Bad Inputs")
    omega = MatrixOmega()
    A = [[1]]
    B = [[1, 1], [1, 1]] # Mismatch
    try:
        omega.compute_product(A, B)
        print("[FAIL] Should have raised error")
    except (ValueError, IndexError):
        print("[PASS] Caught mismatch")

if __name__ == "__main__":
    test_errors()


