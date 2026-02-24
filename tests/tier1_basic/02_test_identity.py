import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

def test_identity():
    print("Tier 1: [02] Identity Matrix Interaction")
    omega = MatrixOmega()
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    res = omega.compute_product(A, I)
    print(f"Result: {res}")
    assert res == [[float(x) for x in row] for row in A]
    print("[PASS]")

if __name__ == "__main__":
    test_identity()


