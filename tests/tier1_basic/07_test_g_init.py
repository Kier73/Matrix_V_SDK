import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import G_SeriesEngine

def test_g_init():
    print("Tier 1: [07] G-Series Engine Initialization")
    engine = G_SeriesEngine(tile_size=4)
    A = [[1.0]*4 for _ in range(4)]
    B = [[1.0]*4 for _ in range(4)]
    res = engine.multiply(A, B)
    print(f"G-Series Result Shape: {len(res)}x{len(res[0])}")
    assert len(res) == 4
    print("[PASS]")

if __name__ == "__main__":
    test_g_init()


