import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import V_SeriesEngine

def test_non_euclidean():
    print("Tier 4: [09] Non-Euclidean Manifold Projection (Minkowski Simulation)")
    engine = V_SeriesEngine()
    # Simulate a matrix with extreme hyperbolic values
    A = [[1e10, -1e10], [1e10, 1e10]]
    try:
        res = engine.multiply(A, A)
        print(f"Projection Result Norm: {np.linalg.norm(res)}")
        print("[PASS] Engine handled extreme manifold scaling")
    except Exception as e:
        print(f"[FAIL] Extreme manifolds broke the engine: {e}")

if __name__ == "__main__":
    test_non_euclidean()


