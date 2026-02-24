import sys
import os
import time
import numpy as np
import math

sys.path.append(os.path.abspath(os.curdir))

from matrix_v_sdk.vl.substrate.anchor import AnchorNavigator
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

def test_anchored_rns_exact_integer():
    print("\n=== [DEPTH] 1. RNS-Exact Precision (Integer Low-Rank Manifold) ===")
    
    # Create an EXACT rank-3 integer matrix
    m, k, n = 50, 100, 50
    r = 3
    # Use SMALL integers to avoid any possible scaling issues for now
    U = np.random.randint(-5, 6, size=(m, r))
    V = np.random.randint(-5, 6, size=(n, r))
    # Intermediate mask
    W_mask = np.random.randint(-2, 3, size=(r, k))
    A = U @ W_mask
    B = np.random.randint(-2, 3, size=(k, r)) @ V.T
    
    # Ground Truth
    C_gt = A @ B
    
    # Initialize Navigator with Exact path, scale=1 for integer work
    # Note: scale=1 means no fractional bits
    nav = AnchorNavigator(A, B, anchor_size=r, exact=True, scale=1)
    print(f"  Navigator: {nav}")
    
    # Spot check elements
    print("  Verifying all elements (Exact Path):")
    C_nav = np.zeros((m, n))
    for ri in range(m):
        for ci in range(n):
            C_nav[ri, ci] = nav.navigate(ri, ci)
    
    max_err = np.max(np.abs(C_nav - C_gt))
    print(f"  Max Error on {m}x{n} grid: {max_err:.2e}")
    if max_err == 0:
        print("  [v] RNS-EXACT INTEGER PATH VERIFIED")
    else:
        print("  [x] RNS-EXACT INTEGER PATH FAILED")

def test_anchored_fp_fidelity():
    print("\n=== [DEPTH] 2. FP Fidelity (SVD-Truncated Low-Rank) ===")
    
    m, k, n = 300, 600, 300
    s = 20
    # Generate random matrix and truncate to rank s
    A_raw = np.random.randn(m, k)
    B_raw = np.random.randn(k, n)
    
    U, sig, Vt = np.linalg.svd(A_raw, full_matrices=False)
    A = U[:, :s] @ np.diag(sig[:s]) @ Vt[:s, :]
    
    U, sig, Vt = np.linalg.svd(B_raw, full_matrices=False)
    B = U[:, :s] @ np.diag(sig[:s]) @ Vt[:s, :]
    
    # C = A @ B should have rank <= s
    C_gt = A @ B
    
    nav = AnchorNavigator(A, B, anchor_size=s)
    print(f"  Navigator: {nav}")
    
    # Verify block recovery
    block_rows = np.random.choice(m, 64, replace=False)
    block_cols = np.random.choice(n, 64, replace=False)
    
    res = nav.navigate_block(block_rows, block_cols)
    gt_sub = C_gt[np.ix_(block_rows, block_cols)]
    
    rel_err = np.linalg.norm(res - gt_sub) / np.linalg.norm(gt_sub)
    print(f"  64x64 Block Relative Error: {rel_err:.2e}")
    if rel_err < 1e-10:
        print("  [v] FP ANCHOR FIDELITY VERIFIED")

def test_anchored_speedup():
    print("\n=== [DEPTH] 3. Speedup Metrics ($O(k)$ vs $O(s^2)$) ===")
    
    m, k, n = 1000, 2000, 1000
    s = 8
    # Just need it to be rank-limited for the math to be fair
    A = np.random.randn(m, s) @ np.random.randn(s, k)
    B = np.random.randn(k, s) @ np.random.randn(s, n)
    
    nav = AnchorNavigator(A, B, anchor_size=s)
    
    Q = 5000 # More queries for better timing
    rows = np.random.randint(0, m, Q)
    cols = np.random.randint(0, n, Q)
    
    # Traditional
    t0 = time.perf_counter()
    # Use numpy dot for fair comparison against nav's vectorization if possible
    # but nav.navigate is scalar, so let's do scalar dot
    for i in range(Q):
        _ = np.dot(A[rows[i], :], B[:, cols[i]])
    lat_trad = (time.perf_counter() - t0) * 1000
    
    # Anchored
    t0 = time.perf_counter()
    for i in range(Q):
        _ = nav.navigate(rows[i], cols[i])
    lat_anch = (time.perf_counter() - t0) * 1000
    
    print(f"  Queries: {Q}")
    print(f"  Traditional O(k) Latency: {lat_trad:.2f}ms")
    print(f"  Anchored O(s^2) Latency:  {lat_anch:.2f}ms")
    print(f"  Realized Speedup: {lat_trad / lat_anch:.2f}x")

if __name__ == "__main__":
    test_anchored_rns_exact_integer()
    test_anchored_fp_fidelity()
    test_anchored_speedup()


