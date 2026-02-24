import os
import sys
import numpy as np

# Add SDK parent to path
SDK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDK_PARENT = os.path.dirname(SDK_ROOT)
sys.path.insert(0, SDK_PARENT)

from matrix_v_sdk.vl.substrate.acceleration import G_SeriesEngine

def audit_tile_parity():
    print("--- PARITY AUDIT: G-SERIES TILE CACHE ---")
    
    A = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    B = [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]
    
    # 1. Standard NumPy
    expected = np.array(A) @ np.array(B)
    
    # 2. G-Series Engine
    engine = G_SeriesEngine(tile_size=2)
    result = engine.multiply(A, B)
    
    print(f"Expected[0,0]: {expected[0,0]}")
    print(f"G-Series[0,0]: {result[0][0]}")
    
    diff = np.abs(expected - np.array(result)).max()
    print(f"\nMax Deviation: {diff:.8f}")
    
    if diff < 1e-6:
        print("[PASS] G-Series Tile Cache is MATEMATICALLY IDENTICAL to Dense Product.")
        print("This is your 'Numerical Memory Wall' hero.")
    else:
        print("[FAIL] G-Series has numerical drift.")

if __name__ == "__main__":
    audit_tile_parity()


