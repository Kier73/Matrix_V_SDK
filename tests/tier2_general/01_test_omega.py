import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

def test_omega():
    print("Tier 2: [01] Strategy Selection Logic")
    omega = MatrixOmega()
    
    # 1. MMP Selection (Rectangular Bottleneck) — needs m*k*n > 262144 AND K > 5*M, K > 5*N
    A_rect = [[1.0]*2000 for _ in range(20)]
    B_rect = [[1.0]*20 for _ in range(2000)]
    s1 = omega.auto_select_strategy(A_rect, B_rect)
    print(f"Strategy for Rectangular: {s1}")
    assert s1 == "mmp", f"Expected 'mmp', got '{s1}'"
    
    # 2. Large Dense Selection (n > 512 -> qmatrix)
    import numpy as np
    A_dense = np.random.rand(600, 600).tolist()
    s2 = omega.auto_select_strategy(A_dense, A_dense)
    print(f"Strategy for Large Dense: {s2}")
    assert s2 == "qmatrix", f"Expected 'qmatrix', got '{s2}'"

    # 3. Cost gate (small matrix -> dense)
    A_small = [[1.0]*10 for _ in range(10)]
    s3 = omega.auto_select_strategy(A_small, A_small)
    print(f"Strategy for Small: {s3}")
    assert s3 == "dense", f"Expected 'dense', got '{s3}'"
    print("[PASS]")

if __name__ == "__main__":
    test_omega()


