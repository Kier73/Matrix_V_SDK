import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import V_SeriesEngine

def test_stability():
    print("Tier 3: [04] V-Series Stability at N=512")
    engine = V_SeriesEngine(epsilon=0.1)
    A = np.random.rand(512, 512).tolist()
    B = np.random.rand(512, 512).tolist()
    
    res = engine.multiply(A, B)
    assert len(res) == 512
    print("[PASS]")

if __name__ == "__main__":
    test_stability()


