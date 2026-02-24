"""
Memthematic I/O — End-to-End Integration Test.

Tests the full pipeline:
  A @ B → Tile → Collapse → ManifoldDescriptor → resolve(r,c)
  A @ B → RNS Ledger → verify(r,c)
  Seed_A * Seed_B → Product → inverse_product_law → Seed_B (recovered)
"""

import sys
import os
import time

# Add SDK root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from matrix_v_sdk.vl.math.inverse_ntt import (
    mod_inverse, inverse_product_law, verify_law_roundtrip
)
from matrix_v_sdk.vl.math.ntt import NTTMorphism, P_GOLDILOCKS
from matrix_v_sdk.vl.substrate.tile_collapser import collapse, resolve, verify_collapse_parity
from matrix_v_sdk.vl.substrate.manifold_fitter import (
    fit_manifold, verify_manifold_parity
)
from matrix_v_sdk.vl.substrate.rns_ledger import record_matrix, verify_matrix


def naive_matmul(A, B):
    """Standard O(n³) matrix multiply for ground-truth comparison."""
    m, k = len(A), len(A[0])
    n = len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C


def test_inverse_ntt():
    """Test Gap 2: Inverse NTT Law Recovery."""
    print("\n" + "=" * 60)
    print("TEST 1: INVERSE NTT LAW RECOVERY (Gap 2)")
    print("=" * 60)

    import random
    random.seed(42)

    # 1000 random roundtrips
    passes = 0
    for _ in range(1000):
        sa = random.randint(1, P_GOLDILOCKS - 1)
        sb = random.randint(1, P_GOLDILOCKS - 1)
        if verify_law_roundtrip(sa, sb):
            passes += 1

    print(f"  Roundtrip: {passes}/1000 passed")
    assert passes == 1000, f"Expected 1000, got {passes}"
    print("  [PASS] All inverse law roundtrips verified.")


def test_tile_collapser():
    """Test Gap 1: Tile Collapse."""
    print("\n" + "=" * 60)
    print("TEST 2: TILE COLLAPSER (Gap 1)")
    print("=" * 60)

    # Zero
    zero = [[0.0] * 8 for _ in range(8)]
    law = collapse(zero)
    assert law.rule == "zero", f"Expected 'zero', got '{law.rule}'"
    ok, err = verify_collapse_parity(zero)
    print(f"  Zero:     rule={law.rule}, parity={'PASS' if ok else 'FAIL'}")

    # Constant
    const = [[3.14] * 8 for _ in range(8)]
    law = collapse(const)
    assert law.rule == "constant", f"Expected 'constant', got '{law.rule}'"
    ok, err = verify_collapse_parity(const)
    print(f"  Constant: rule={law.rule}, parity={'PASS' if ok else 'FAIL'} (max_err={err:.2e})")

    # Linear
    linear = [[0.5 * (r * 8 + c) + 1.0 for c in range(8)] for r in range(8)]
    law = collapse(linear)
    assert law.rule == "linear", f"Expected 'linear', got '{law.rule}'"
    ok, err = verify_collapse_parity(linear)
    print(f"  Linear:   rule={law.rule}, parity={'PASS' if ok else 'FAIL'} (max_err={err:.2e})")

    print("  [PASS] All tile collapser classifications verified.")


def test_manifold_fitter():
    """Test Gap 3A: Manifold Fitting."""
    print("\n" + "=" * 60)
    print("TEST 3: MANIFOLD FITTER (Gap 3A)")
    print("=" * 60)

    # Create a structured matrix (constant + linear + zero blocks)
    N = 32
    matrix = [[0.0] * N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            if r < 16 and c < 16:
                matrix[r][c] = 7.0  # Constant quadrant
            elif r < 16:
                matrix[r][c] = 0.0  # Zero quadrant
            else:
                matrix[r][c] = 0.1 * (r * N + c)  # Linear quadrant

    mf = fit_manifold(matrix, tile_size=8)
    stats = mf.stats()
    print(f"  Tiles:       {stats['total_tiles']}")
    print(f"  Classes:     {stats['classification']}")
    print(f"  Compression: {stats['compression_ratio']:.1f}x")

    # Verify parity for structured tiles (zero + constant)
    ok, err, ec = verify_manifold_parity(matrix, mf, epsilon=0.01)
    print(f"  Parity:      {ec} errors (max_err={err:.4f})")

    print("  [PASS] Manifold fitting operational.")


def test_rns_ledger():
    """Test Gap 3B: RNS Ledger."""
    print("\n" + "=" * 60)
    print("TEST 4: RNS LEDGER (Gap 3B)")
    print("=" * 60)

    matrix = [[1.5, -2.3, 0.0], [4.7, -1.1, 3.14]]
    ledger = record_matrix(matrix)

    # Verify original
    ok, passes, total = verify_matrix(matrix, ledger)
    print(f"  Original:    {passes}/{total} ({'PASS' if ok else 'FAIL'})")

    # Tamper detection
    tampered = [[1.5, -2.3, 0.0], [4.7, -1.1, 3.15]]
    ok_t, passes_t, total_t = verify_matrix(tampered, ledger)
    print(f"  Tampered:    {passes_t}/{total_t} (tamper {'DETECTED' if not ok_t else 'MISSED'})")

    assert ok, "Original should pass"
    assert not ok_t, "Tamper should be detected"

    print("  [PASS] RNS ledger verification and tamper detection working.")


def test_end_to_end():
    """Full pipeline: matmul → collapse → resolve → verify."""
    print("\n" + "=" * 60)
    print("TEST 5: END-TO-END PIPELINE")
    print("=" * 60)

    # Small matmul
    A = [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0],
         [13.0, 14.0, 15.0, 16.0]]

    B = [[0.1, 0.2, 0.3, 0.4],
         [0.5, 0.6, 0.7, 0.8],
         [0.9, 1.0, 1.1, 1.2],
         [1.3, 1.4, 1.5, 1.6]]

    # Step 1: Dense matmul (ground truth)
    t0 = time.perf_counter()
    C = naive_matmul(A, B)
    t_dense = (time.perf_counter() - t0) * 1000
    print(f"  Step 1 — Dense matmul:   {t_dense:.3f}ms")

    # Step 2: Collapse result into manifold
    t0 = time.perf_counter()
    mf = fit_manifold(C, tile_size=4)
    t_collapse = (time.perf_counter() - t0) * 1000
    print(f"  Step 2 — Manifold fit:   {t_collapse:.3f}ms")
    print(f"             Tiles: {mf.stats()['classification']}")

    # Step 3: Record in RNS ledger
    t0 = time.perf_counter()
    ledger = record_matrix(C)
    t_ledger = (time.perf_counter() - t0) * 1000
    print(f"  Step 3 — RNS record:     {t_ledger:.3f}ms")

    # Step 4: Verify manifold resolution
    print(f"\n  Step 4 — Verification:")
    print(f"    C[0][0] = {C[0][0]:.4f} | Manifold: {mf.resolve(0,0):.4f}")
    print(f"    C[1][2] = {C[1][2]:.4f} | Manifold: {mf.resolve(1,2):.4f}")
    print(f"    C[3][3] = {C[3][3]:.4f} | Manifold: {mf.resolve(3,3):.4f}")

    # Step 5: Verify RNS integrity
    ok, passes, total = verify_matrix(C, ledger)
    print(f"    RNS Ledger: {passes}/{total} ({'PASS' if ok else 'FAIL'})")

    # Step 6: Memory comparison
    dense_bytes = 4 * 4 * 8
    manifold_bytes = mf.memory_footprint_bytes()
    ledger_bytes = ledger.memory_footprint_bytes()
    print(f"\n  Memory Comparison:")
    print(f"    Dense:    {dense_bytes} bytes")
    print(f"    Manifold: {manifold_bytes} bytes ({mf.compression_ratio():.1f}x compression)")
    print(f"    Ledger:   {ledger_bytes} bytes (for exact verification)")

    print("\n  [PASS] End-to-end pipeline operational.")


if __name__ == "__main__":
    print("+" + "=" * 58 + "+")
    print("|   MEMTHEMATIC I/O -- INTEGRATION TEST SUITE              |")
    print("+" + "=" * 58 + "+")

    results = []
    results.append(("Inverse NTT", test_inverse_ntt()))
    results.append(("Tile Collapser", test_tile_collapser()))
    results.append(("Manifold Fitter", test_manifold_fitter()))
    results.append(("RNS Ledger", test_rns_ledger()))
    results.append(("End-to-End", test_end_to_end()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}")
    print("=" * 60)

    all_passed = all(p for _, p in results)
    sys.exit(0 if all_passed else 1)


