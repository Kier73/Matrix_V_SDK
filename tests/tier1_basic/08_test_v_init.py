import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import V_SeriesEngine

def test_v_init():
    print("Tier 1: [08] V-Series Engine Initialization")
    engine = V_SeriesEngine(epsilon=0.1)
    # Check adaptive D for small dimension
    d = engine.get_adaptive_d(10)
    print(f"Adaptive D for N=10: {d}")
    assert d > 0
    print("[PASS]")

if __name__ == "__main__":
    test_v_init()


