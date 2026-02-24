import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import RH_SeriesEngine

def test_rh_init():
    print("Tier 1: [10] RH-Series Engine Initialization")
    engine = RH_SeriesEngine()
    is_p = RH_SeriesEngine.is_prime(17)
    print(f"Is 17 prime? {is_p}")
    assert is_p == True
    print("[PASS]")

if __name__ == "__main__":
    test_rh_init()


