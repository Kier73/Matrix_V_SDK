import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

def test_dot():
    print("Tier 1: [01] Dot Product Correctness")
    omega = MatrixOmega()
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    # Expected: [[19, 22], [43, 50]]
    res = omega.compute_product(A, B)
    print(f"Result: {res}")
    assert res == [[19.0, 22.0], [43.0, 50.0]]
    print("[PASS]")

if __name__ == "__main__":
    test_dot()


