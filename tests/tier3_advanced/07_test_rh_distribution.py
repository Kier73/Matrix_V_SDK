import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import RH_SeriesEngine

def test_rh_dist():
    print("Tier 3: [07] Mobius Density Verification (N=1000)")
    rh = RH_SeriesEngine()
    count_nonzero = 0
    for i in range(1, 1001):
        if rh.get_mobius(i) != 0:
            count_nonzero += 1
    
    density = count_nonzero / 1000.0
    print(f"Square-free density: {density}")
    # Theoretical: 6/pi^2 approx 0.6079
    assert 0.5 < density < 0.7
    print("[PASS]")

if __name__ == "__main__":
    test_rh_dist()


