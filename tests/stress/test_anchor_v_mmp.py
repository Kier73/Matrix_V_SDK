"""
Benchmark: RNS-Exact Anchor vs MMP_Engine
=========================================
Compares the performance of the O(s^2) arithmetic navigation path
against the O(k) MMP (Prime Channelling) path.

SCENARIO:
  N=128, k=1024.
  Matrix is rank-16 (s=16).
  Comparison of time-per-element for exact recovery.

Run: python tests/stress/test_anchor_v_mmp.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from matrix_v_sdk.vl.substrate.anchor import AnchorNavigator
from matrix_v_sdk.vl.substrate.acceleration import MMP_Engine

def benchmark_exact_paths():
    print("\n" + "="*60)
    print("BENCHMARK: RNS-Exact Anchor vs MMP_Engine")
    print("="*60)
    
    m, n, k = 64, 64, 1024
    s = 16
    np.random.seed(42)
    
    # Construct a rank-s matrix
    A_base = np.random.randint(-5, 6, size=(m, s)).astype(float)
    B_base = np.random.randint(-5, 6, size=(s, k)).astype(float)
    A = A_base @ (np.random.rand(s, k) * 0.1) # Mix but keep rank low
    A = A_base @ np.eye(s, k) # Exact rank-s
    
    B = np.random.randint(-5, 6, size=(k, n)).astype(float)
    
    print(f"Matrix: {m}x{n} (inner k={k}), Target Rank s={s}")
    
    # 1. MMP_Engine (O(k) per element)
    print("\n[MMP Engine] Running...")
    mmp = MMP_Engine()
    start = time.time()
    C_mmp = mmp.multiply(A.tolist(), B.tolist())
    dur_mmp = time.time() - start
    print(f"  Duration: {dur_mmp:.4f}s")
    
    # 2. RNS-Exact Anchor (O(s^2) per element)
    print("\n[Anchor Navigator (Exact)] Initializing...")
    start_init = time.time()
    nav = AnchorNavigator(A, B, anchor_size=s, exact=True, scale=1)
    dur_init = time.time() - start_init
    print(f"  Init Time (Anchor + Inv): {dur_init:.4f}s")
    
    print("  Navigating all elements...")
    start_nav = time.time()
    C_nav = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(nav.navigate(i, j))
        C_nav.append(row)
    dur_nav = time.time() - start_nav
    print(f"  Navigation Time: {dur_nav:.4f}s")
    print(f"  Total Time: {dur_init + dur_nav:.4f}s")
    
    # Speedup
    total_anchor = dur_init + dur_nav
    speedup = dur_mmp / total_anchor
    print(f"\nSPEEDUP (Anchor vs MMP): {speedup:.2f}x")
    
    # Verification
    diff = np.abs(np.array(C_mmp) - np.array(C_nav))
    max_diff = np.max(diff)
    print(f"Parity Check (Max Diff): {max_diff:.2e}")
    
    if max_diff == 0:
        print("[SUCCESS] Both engines produced identical exact results.")
    else:
        print("[NOTE] Engines differ slightly (check scale/quantization).")

if __name__ == "__main__":
    benchmark_exact_paths()


