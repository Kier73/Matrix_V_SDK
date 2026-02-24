import os
import sys
import time
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

def test_medium_dense():
    print("Tier 2: [09] Medium Dense Performance (N=256)")
    omega = MatrixOmega()
    A = np.random.rand(256, 256).tolist()
    B = np.random.rand(256, 256).tolist()
    
    start = time.time()
    omega.compute_product(A, B)
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.2f}s")
    print("[PASS]")

if __name__ == "__main__":
    test_medium_dense()


