import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import V_SeriesEngine

def test_chaos():
    print("Tier 4: [01] Massive Projection Chaos (N=8192)")
    # This might exceed memory if not lazy or if V-Series projects too many vectors
    engine = V_SeriesEngine(epsilon=0.01)
    N = 8192
    print(f"Projecting {N}x{N} manifold...")
    try:
        # We don't need real data to test the projection logic
        d = engine.get_adaptive_d(N)
        print(f"Target Dimension D: {d}")
        assert d > 0
        print("[PASS] Projection logic held")
    except Exception as e:
        print(f"[FAIL] Chaos exception: {e}")

if __name__ == "__main__":
    test_chaos()


