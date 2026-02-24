"""
Memthematic I/O -- Limit Stress Test
=====================================
Pushes the full pipeline across multiple dimensions to understand limits:

  1. SCALE:     Matrix size (16 -> 512) — where does collapse quality degrade?
  2. STRUCTURE:  Zero / Constant / Linear / Random / Sparse — which laws collapse best?
  3. PIPELINE:   QMatrix.memthematic_multiply end-to-end
  4. INVERSE:    NTT law recovery at scale
  5. COMPRESSION: Manifold size vs dense size

Run:  python tests/stress/test_memthematic_limits.py
"""

import sys
import os
import time
import random
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from matrix_v_sdk.vl.math.inverse_ntt import mod_inverse, inverse_product_law, verify_law_roundtrip
from matrix_v_sdk.vl.math.ntt import NTTMorphism, P_GOLDILOCKS
from matrix_v_sdk.vl.substrate.tile_collapser import collapse, resolve, verify_collapse_parity, TileLaw
from matrix_v_sdk.vl.substrate.manifold_fitter import fit_manifold, verify_manifold_parity, ManifoldDescriptor
from matrix_v_sdk.vl.substrate.rns_ledger import record_matrix, verify_matrix, RNSLedger


# ─── Helpers ──────────────────────────────────────────────

def naive_matmul(A, B):
    """Standard O(n^3) matmul."""
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


def make_matrix(n, kind="random", seed=42):
    """Generate an n x n test matrix of a given kind."""
    rng = random.Random(seed)
    if kind == "zero":
        return [[0.0] * n for _ in range(n)]
    elif kind == "constant":
        val = 3.14159
        return [[val] * n for _ in range(n)]
    elif kind == "linear":
        return [[0.01 * (r * n + c) for c in range(n)] for r in range(n)]
    elif kind == "sparse":
        # 5% nonzero
        mat = [[0.0] * n for _ in range(n)]
        for _ in range(max(1, n * n // 20)):
            r, c = rng.randint(0, n-1), rng.randint(0, n-1)
            mat[r][c] = rng.gauss(0, 1)
        return mat
    elif kind == "identity":
        mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            mat[i][i] = 1.0
        return mat
    else:  # random
        return [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]


def count_laws(mf: ManifoldDescriptor) -> dict:
    counts = {"zero": 0, "constant": 0, "linear": 0, "complex": 0}
    for entry in mf.tiles:
        counts[entry.law.rule] += 1
    return counts


# ─── Test 1: Scale Limits ─────────────────────────────────

def test_scale():
    """How does collapse quality scale with matrix size?"""
    print("\n" + "=" * 70)
    print("TEST 1: SCALE LIMITS (collapse quality vs matrix size)")
    print("=" * 70)

    sizes = [16, 32, 64, 128, 256]
    tile_size = 16

    print(f"  {'N':>6} | {'Tile':>4} | {'Tiles':>5} | {'Compress':>8} | "
          f"{'Laws':>20} | {'Time':>8}")
    print("  " + "-" * 66)

    for n in sizes:
        A = make_matrix(n, "random", seed=1)
        B = make_matrix(n, "random", seed=2)

        t0 = time.perf_counter()
        C = naive_matmul(A, B)
        mf = fit_manifold(C, tile_size=tile_size)
        elapsed = (time.perf_counter() - t0) * 1000

        laws = count_laws(mf)
        cr = mf.compression_ratio()

        law_str = f"z:{laws['zero']} c:{laws['constant']} l:{laws['linear']} x:{laws['complex']}"
        print(f"  {n:>6} | {tile_size:>4} | {len(mf.tiles):>5} | "
              f"{cr:>7.1f}x | {law_str:>20} | {elapsed:>7.1f}ms")

    print("  [DONE] Scale test complete.")


# ─── Test 2: Structure Limits ─────────────────────────────

def test_structure():
    """Which data structures collapse best?"""
    print("\n" + "=" * 70)
    print("TEST 2: STRUCTURE LIMITS (collapse quality by data type)")
    print("=" * 70)

    n = 64
    tile_size = 16
    kinds = ["zero", "constant", "linear", "identity", "sparse", "random"]

    print(f"  {'Kind':>10} | {'Compress':>8} | {'Laws':>25} | "
          f"{'Exact':>5} | {'MaxErr':>10}")
    print("  " + "-" * 68)

    for kind in kinds:
        A = make_matrix(n, kind, seed=10)
        B = make_matrix(n, kind, seed=20)
        C = naive_matmul(A, B)

        mf = fit_manifold(C, tile_size=tile_size)
        laws = count_laws(mf)
        cr = mf.compression_ratio()

        # Check parity for structured results
        ok, max_err, ec = verify_manifold_parity(C, mf, epsilon=1e-6)

        law_str = f"z:{laws['zero']} c:{laws['constant']} l:{laws['linear']} x:{laws['complex']}"
        exact_str = "YES" if ok else f"{ec} err"
        print(f"  {kind:>10} | {cr:>7.1f}x | {law_str:>25} | "
              f"{exact_str:>5} | {max_err:>10.2e}")

    print("  [DONE] Structure test complete.")


# ─── Test 3: RNS Ledger Limits ────────────────────────────

def test_rns_limits():
    """RNS ledger accuracy across value ranges."""
    print("\n" + "=" * 70)
    print("TEST 3: RNS LEDGER LIMITS (value range vs accuracy)")
    print("=" * 70)

    n = 32
    ranges = [
        ("Tiny [-0.01, 0.01]", 0.01),
        ("Small [-1, 1]", 1.0),
        ("Medium [-100, 100]", 100.0),
        ("Large [-10000, 10000]", 10000.0),
        ("Huge [-1e6, 1e6]", 1e6),
    ]

    print(f"  {'Range':>25} | {'Pass':>6} | {'Total':>5} | {'Rate':>6}")
    print("  " + "-" * 50)

    rng = random.Random(42)
    for label, r in ranges:
        matrix = [[rng.uniform(-r, r) for _ in range(n)] for _ in range(n)]
        ledger = record_matrix(matrix, scale=1e6)
        ok, passes, total = verify_matrix(matrix, ledger)
        rate = passes / total * 100
        print(f"  {label:>25} | {passes:>6} | {total:>5} | {rate:>5.1f}%")

    print("  [DONE] RNS ledger limits verified.")


# ─── Test 4: Inverse NTT at Scale ────────────────────────

def test_inverse_ntt_scale():
    """Inverse NTT law recovery across seed ranges."""
    print("\n" + "=" * 70)
    print("TEST 4: INVERSE NTT LAW RECOVERY (scale + edge cases)")
    print("=" * 70)

    rng = random.Random(42)
    test_cases = [
        ("Small seeds (< 1000)", [(rng.randint(1, 999), rng.randint(1, 999)) for _ in range(1000)]),
        ("Medium seeds (< 2^32)", [(rng.randint(1, 2**32), rng.randint(1, 2**32)) for _ in range(1000)]),
        ("Large seeds (< 2^63)", [(rng.randint(1, 2**63), rng.randint(1, 2**63)) for _ in range(1000)]),
        ("Near-prime seeds", [(P_GOLDILOCKS - rng.randint(1, 100), P_GOLDILOCKS - rng.randint(1, 100)) for _ in range(1000)]),
    ]

    print(f"  {'Category':>25} | {'Pass':>6} | {'Total':>5} | {'Rate':>6} | {'Time':>8}")
    print("  " + "-" * 60)

    for label, pairs in test_cases:
        t0 = time.perf_counter()
        passes = sum(1 for a, b in pairs if verify_law_roundtrip(a, b))
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  {label:>25} | {passes:>6} | {len(pairs):>5} | "
              f"{passes/len(pairs)*100:>5.1f}% | {elapsed:>7.1f}ms")

    print("  [DONE] Inverse NTT scale test complete.")


# ─── Test 5: QMatrix Pipeline ────────────────────────────

def test_qmatrix_pipeline():
    """Full QMatrix.memthematic_multiply pipeline."""
    print("\n" + "=" * 70)
    print("TEST 5: QMATRIX MEMTHEMATIC PIPELINE (end-to-end)")
    print("=" * 70)

    from matrix_v_sdk.vl.substrate.unified import QMatrix

    sizes = [16, 32, 64, 128]

    print(f"  {'N':>6} | {'Result':>40} | {'Time':>8}")
    print("  " + "-" * 60)

    for n in sizes:
        A = make_matrix(n, "random", seed=100)
        B = make_matrix(n, "random", seed=200)

        t0 = time.perf_counter()
        q = QMatrix(seed=0x42, tile_size=min(n, 32))
        result = q.memthematic_multiply(A, B, verify=True)
        elapsed = (time.perf_counter() - t0) * 1000

        # Spot-check: resolve a few elements and verify against ledger
        C_dense = naive_matmul(A, B)
        checks = 0
        rns_ok = 0
        for i in range(min(10, n)):
            for j in range(min(10, n)):
                if result.verify(i, j, C_dense[i][j]):
                    rns_ok += 1
                checks += 1

        print(f"  {n:>6} | {repr(result):>40} | {elapsed:>7.1f}ms")
        print(f"         | RNS verify: {rns_ok}/{checks} spot-checks passed")

    print("  [DONE] QMatrix pipeline test complete.")


# ─── Test 6: Compression Frontier ────────────────────────

def test_compression_frontier():
    """Map the compression ratio across tile sizes and data types."""
    print("\n" + "=" * 70)
    print("TEST 6: COMPRESSION FRONTIER (tile_size vs compression)")
    print("=" * 70)

    n = 128
    tile_sizes = [8, 16, 32, 64]
    kinds = ["zero", "constant", "linear", "sparse", "random"]

    # Header
    header = f"  {'Kind':>10} |"
    for T in tile_sizes:
        header += f" T={T:>3} |"
    print(header)
    print("  " + "-" * (14 + 8 * len(tile_sizes)))

    for kind in kinds:
        A = make_matrix(n, kind, seed=10)
        B = make_matrix(n, kind, seed=20)
        C = naive_matmul(A, B)

        row = f"  {kind:>10} |"
        for T in tile_sizes:
            mf = fit_manifold(C, tile_size=T)
            cr = mf.compression_ratio()
            row += f" {cr:>5.1f}x|"
        print(row)

    print("  [DONE] Compression frontier mapped.")


# ─── Main ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("+" + "=" * 68 + "+")
    print("|  MEMTHEMATIC I/O -- LIMIT STRESS TEST                              |")
    print("|  Testing collapse quality, RNS integrity, and pipeline at scale    |")
    print("+" + "=" * 68 + "+")

    t_total = time.perf_counter()

    test_scale()
    test_structure()
    test_rns_limits()
    test_inverse_ntt_scale()
    test_qmatrix_pipeline()
    test_compression_frontier()

    total_time = time.perf_counter() - t_total

    print("\n" + "=" * 70)
    print(f"  TOTAL TIME: {total_time:.2f}s")
    print("=" * 70)


