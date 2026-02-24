import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

def test_scalar():
    print("Tier 1: [05] Scalar Multiplication Simulation")
    omega = MatrixOmega()
    A = [[2, 2], [2, 2]]
    S = [[5, 0], [0, 5]]
    res = omega.compute_product(A, S)
    print(f"Result: {res}")
    assert res == [[10.0, 10.0], [10.0, 10.0]]
    print("[PASS]")

if __name__ == "__main__":
    test_scalar()


