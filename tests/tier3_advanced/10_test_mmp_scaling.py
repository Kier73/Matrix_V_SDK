import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import MMP_Engine

def test_mmp_scaling():
    print("Tier 3: [10] Rectangular Bottleneck Scaling (MMP)")
    engine = MMP_Engine()
    A = [[1.0]*1000 for _ in range(16)] # 16x1000
    B = [[1.0]*16 for _ in range(1000)] # 1000x16
    
    start = time.time()
    res = engine.multiply(A, B)
    elapsed = time.time() - start
    print(f"MMP Sparse Pass: {elapsed:.4f}s")
    assert len(res) == 16
    print("[PASS]")

if __name__ == "__main__":
    test_mmp_scaling()


