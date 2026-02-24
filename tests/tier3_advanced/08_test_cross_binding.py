import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import X_SeriesEngine

def test_cross():
    print("Tier 3: [08] X-Series + Manifold Binding")
    x1 = X_SeriesEngine(seed=0x1)
    x2 = X_SeriesEngine(seed=0x2)
    x3 = x1.bind(x2)
    assert x3.seed == 0x1 ^ 0x2
    print("[PASS]")

if __name__ == "__main__":
    test_cross()


