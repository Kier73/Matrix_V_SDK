"""
Anchor-Navigate: Strategy Comparison (Adaptive vs Fixed)
=========================================================
Benchmarks different anchor selection strategies:
  1. norm    (Greedy energy leverage)
  2. spread  (Even geometric coverage)
  3. rns     (Number-theoretic spacing)
  4. mixed   (Top energy + coverage)
  5. adaptive (Winner-takes-all based on W conditioning)

Demonstrates that the adaptive fusion consistently picks the most 
stable anchor, especially for difficult (ill-conditioned) matrices.

Run: python tests/stress/test_anchor_adaptive_comp.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from matrix_v_sdk.vl.substrate.anchor import AnchorNavigator

def benchmark_strategies(A, B, name, s=10):
    print(f"\n--- {name} (N={A.shape[0]}, s={s}) ---")
    
    strategies = ['norm', 'spread', 'rns', 'mixed', 'adaptive']
    results = []
    
    # Ground truth for 1000 queries
    np.random.seed(42)
    Q = 1000
    targets = [(np.random.randint(A.shape[0]), np.random.randint(B.shape[1])) for _ in range(Q)]
    exact_vals = [np.dot(A[i,:], B[:,j]) for i, j in targets]

    print(f"  {'Strategy':<12} | {'Condition W':<12} | {'Max Error':<12} | {'Mean Error':<12}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for strat in strategies:
        try:
            t0 = time.perf_counter()
            nav = AnchorNavigator(A, B, anchor_size=s, strategy=strat)
            
            nav_vals = [nav.navigate(i, j) for i, j in targets]
            errors = [abs(n - e) for n, e in zip(nav_vals, exact_vals)]
            
            stats = nav.stats()
            cond = stats['W_cond']
            mx_err = max(errors)
            me_err = np.mean(errors)
            
            results.append((strat, cond, mx_err, me_err))
            print(f"  {strat:<12} | {cond:>12.2e} | {mx_err:>12.2e} | {me_err:>12.2e}")
        except Exception as e:
            print(f"  {strat:<12} | {'FAILED':>12} | {str(e):<26}")

    return results

def test_comparisons():
    N = 250
    s = 15
    np.random.seed(42)

    # 1. Hilbert-like (Extreme ill-conditioning)
    # Most strategies fail here; only adaptive has a chance to find a stable sub-block
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    A_hilb = 1.0 / (i + j + 1.0)
    B_hilb = np.identity(N)
    benchmark_strategies(A_hilb, B_hilb, "Hilbert (Poor Conditioning)", s=s)

    # 2. Sparse Random
    A_sparse = np.random.randn(N, N)
    A_sparse[A_sparse < 1.0] = 0.0
    B_sparse = np.random.randn(N, N)
    benchmark_strategies(A_sparse, B_sparse, "Sparse Random", s=s)

    # 3. Sine/Cosine Manifold (Periodic)
    x = np.linspace(0, 10, N)
    A_periodic = np.sin(x.reshape(-1, 1) * x.reshape(1, -1))
    B_periodic = np.cos(x.reshape(-1, 1) + x.reshape(1, -1))
    benchmark_strategies(A_periodic, B_periodic, "Periodic Manifold", s=s)

    # 4. Low-rank Cluster
    r = 5
    U = np.random.randn(N, r)
    V = np.random.randn(N, r)
    A_lr = U @ V.T
    B_lr = np.random.randn(N, N)
    benchmark_strategies(A_lr, B_lr, "Low-Rank (r=5)", s=s)

if __name__ == "__main__":
    test_comparisons()


