import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

def test_zeros():
    print("Tier 1: [03] Zero Matrix Interaction")
    omega = MatrixOmega()
    A = [[1, 2], [3, 4]]
    Z = [[0, 0], [0, 0]]
    res = omega.compute_product(A, Z)
    print(f"Result: {res}")
    assert res == [[0.0, 0.0], [0.0, 0.0]]
    print("[PASS]")

if __name__ == "__main__":
    test_zeros()


