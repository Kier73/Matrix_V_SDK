import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import X_SeriesEngine

def test_x_init():
    print("Tier 1: [09] X-Series Engine Initialization")
    engine = X_SeriesEngine(seed=0x123)
    val = engine.resolve_element(0, 0)
    print(f"X-Series Resolve (0,0): {val}")
    assert val in [1.0, -1.0]
    print("[PASS]")

if __name__ == "__main__":
    test_x_init()


