import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

def test_rect():
    print("Tier 1: [04] Rectangular Matrix Correctness")
    omega = MatrixOmega()
    A = [[1, 2, 3], [4, 5, 6]] # 2x3
    B = [[7, 8], [9, 10], [11, 12]] # 3x2
    # Product: 2x2
    # [1*7+2*9+3*11, 1*8+2*10+3*12] = [7+18+33, 8+20+36] = [58, 64]
    # [4*7+5*9+6*11, 4*8+5*10+6*12] = [28+45+66, 32+50+72] = [139, 154]
    res = omega.compute_product(A, B)
    print(f"Result: {res}")
    assert res == [[58.0, 64.0], [139.0, 154.0]]
    print("[PASS]")

if __name__ == "__main__":
    test_rect()


