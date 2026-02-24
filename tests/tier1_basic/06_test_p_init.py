import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import P_SeriesEngine

def test_p_init():
    print("Tier 1: [06] P-Series Engine Initialization")
    engine = P_SeriesEngine()
    val = P_SeriesEngine.resolve_p_series(2, 4, 3) # (2*4) % 3 = 2 != 0
    print(f"P-Series Resolve (2,4,3): {val}")
    assert val == 3.0
    print("[PASS]")

if __name__ == "__main__":
    test_p_init()


