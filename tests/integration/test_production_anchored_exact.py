"""
Production Integration: RNS-Exact Anchor Dispatch
==================================================
Verifies that MatrixOmega correctly promotes dense operations
to 'anchored_exact' when structural redundancy is detected.

THEORY:
  For a dense matrix where k is large but rank s is small, 
  MMP (O(k)) is exact, but Anchored-Exact (O(s^2)) is much faster.
  MatrixOmega should automatically detect this and switch.

Run: python tests/integration/test_production_anchored_exact.py
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

def test_production_exact_dispatch():
    print("\n" + "="*60)
    print("TEST: Production RNS-Exact Dispatch (MatrixOmega)")
    print("="*60)
    
    # Setup Omega
    omega = MatrixOmega(seed=42)
    
    # Case: Dense, Large Inner K, but Low Rank
    # 64 x 1024 * 1024 x 64 (k=1024, s=10)
    m, k, n = 64, 1024, 64
    s = 10
    
    print(f"Scenario: {m}x{k} @ {k}x{n}, rank={s}")
    
    # A and B must be exactly low-rank and periodic to trigger 'anchored_exact'
    tile_size = 4
    A_tile = np.random.randint(-5, 6, size=(m, tile_size)).astype(float)
    # Replicate tile to create periodicity
    A = np.hstack([A_tile] * (k // tile_size))
    
    B_base = np.random.randint(-5, 6, size=(tile_size, n)).astype(float)
    B = np.vstack([B_base] * (k // tile_size)) / (k // tile_size)
    
    # Ground Truth: C = A @ B. 
    # Since A and B are periodic tiles, A @ B = (k/tile) * (A_tile @ B_base) / (k/tile) = A_tile @ B_base
    C_gt = A_tile @ B_base
    
    # Ground Truth (exact integer if we keep values small)
    C_gt = A @ B
    
    # Check Strategy
    strategy = omega.auto_select_strategy(A, B)
    print(f"  Selected Strategy: {strategy}")
    
    # EXPECTED: anchored_exact (because k > 256 and m < 128)
    if strategy == "anchored_exact":
        print("  [PASS] Correct engine promoted for high-fidelity dense manifold.")
    else:
        print(f"  [FAIL] Expected anchored_exact, got {strategy}")
        sys.exit(1)
        
    # Check Result
    print("  Executing compute_product...")
    C_omega = omega.compute_product(A, B)
    
    diff = np.abs(np.array(C_omega) - C_gt)
    max_err = np.max(diff)
    print(f"  Max Error: {max_err:.2e}")
    
    if max_err < 1e-12:
        print("  [PASS] Result matches ground truth with high fidelity.")
    else:
        print("  [FAIL] Result outside precision bounds.")
        sys.exit(1)

def test_fallback_behavior():
    print("\n" + "="*60)
    print("TEST: Fallback to Dense (Small Matrix)")
    print("="*60)
    
    omega = MatrixOmega(seed=42)
    A = np.random.rand(10, 10).tolist()
    B = np.random.rand(10, 10).tolist()
    
    strategy = omega.auto_select_strategy(A, B)
    print(f"  Small Matrix Strategy: {strategy}")
    if strategy == "dense":
        print("  [PASS] Correct fallback to dense for sub-threshold costs.")
    else:
        print(f"  [FAIL] Expected dense, got {strategy}")
        sys.exit(1)

if __name__ == "__main__":
    test_production_exact_dispatch()
    test_fallback_behavior()


