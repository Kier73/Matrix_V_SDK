import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import V_SeriesEngine

def test_min_d():
    print("Tier 4: [10] Pushing Adaptive D to Theoretical Floor")
    engine = V_SeriesEngine()
    for n in [1, 2, 4, 8, 16]:
        d = engine.get_adaptive_d(n)
        print(f"N={n} -> D={d}")
        assert d >= 1
    print("[PASS]")

if __name__ == "__main__":
    test_min_d()


