import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

def test_precision():
    print("Tier 2: [08] Precision Audit (N=128)")
    omega = MatrixOmega()
    A = np.random.rand(128, 128).tolist()
    B = np.random.rand(128, 128).tolist()
    
    res = omega.compute_product(A, B)
    # Compare with numpy
    expected = np.dot(np.array(A), np.array(B))
    diff = np.abs(np.array(res) - expected).max()
    print(f"Max Diff: {diff}")
    assert diff < 1e-4
    print("[PASS]")

if __name__ == "__main__":
    test_precision()


