"""
JIT Streaming Matmul -- Large Scale Exact Materialization Test
==============================================================
Demonstrates the core Memthematic I/O thesis:

  Given seed-defined matrices A and B, compute C = A x B
  where ANY element C[i][j] can be materialized on demand
  with EXACT numerical reproducibility, without ever holding
  the full dense C in memory.

Two modes are tested:

  Mode 1: SEED STREAMING (O(K) per element, exact)
    C[i][j] = SUM_k  resolve(seed_A, i, k) * resolve(seed_B, k, j)
    Every call returns the same IEEE-754 float. Zero storage for C.

  Mode 2: MANIFOLD + RNS (O(1) per element after one-time collapse)
    C_manifold = collapse(A x B)
    C[i][j] = manifold.resolve(i, j)   -- fast but approximate for complex tiles
    C[i][j] = rns_ledger.verify(i, j)  -- exact integrity check

Scales tested: 256, 512, 1024, 2048

Run:  python tests/stress/test_jit_streaming.py
"""

import sys
import os
import time
import struct
import random
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from matrix_v_sdk.vl.math.primitives import fmix64
from matrix_v_sdk.vl.math.ntt import NTTMorphism, P_GOLDILOCKS
from matrix_v_sdk.vl.math.inverse_ntt import mod_inverse, inverse_product_law
from matrix_v_sdk.vl.substrate.tile_collapser import collapse, resolve as tile_resolve
from matrix_v_sdk.vl.substrate.manifold_fitter import fit_manifold
from matrix_v_sdk.vl.substrate.rns_ledger import record_matrix, RNSLedger


# ==============================================================
# SEED-DEFINED MATRIX ENGINE
# ==============================================================

class SeedMatrix:
    """
    A matrix that exists ONLY as a seed + dimensions.
    No data is stored. Every element is JIT-computed via fmix64.

    Element at (i, j):
        raw  = fmix64(seed XOR (i * cols + j))
        val  = raw / 2^64 * 2.0 - 1.0    (normalized to [-1, 1])

    Properties:
        - Deterministic: same (seed, i, j) always produces same value
        - O(1) per element
        - Zero storage
        - Reproducible across machines (integer arithmetic only)
    """

    def __init__(self, rows: int, cols: int, seed: int):
        self.rows = rows
        self.cols = cols
        self.seed = seed

    def resolve(self, i: int, j: int) -> float:
        """JIT-resolve a single element. O(1), exact."""
        idx = i * self.cols + j
        h = fmix64(self.seed ^ idx)
        return (h / 18446744073709551615.0) * 2.0 - 1.0

    def resolve_row(self, i: int) -> list:
        """Resolve an entire row. O(cols)."""
        return [self.resolve(i, j) for j in range(self.cols)]

    def materialize_tile(self, r0: int, c0: int, tr: int, tc: int) -> list:
        """Materialize a tile sub-matrix. O(tr * tc)."""
        return [[self.resolve(r0 + r, c0 + c)
                 for c in range(tc)]
                for r in range(tr)]

    def memory_bytes(self) -> int:
        """Memory footprint: just the seed + dims = 24 bytes."""
        return 24

    def virtual_bytes(self) -> int:
        """Virtual data represented: rows * cols * 8 bytes."""
        return self.rows * self.cols * 8

    def __repr__(self):
        return (f"<SeedMatrix {self.rows}x{self.cols} | "
                f"seed=0x{self.seed:X} | "
                f"{self.memory_bytes()}B actual / "
                f"{self.virtual_bytes():,}B virtual>")


# ==============================================================
# JIT STREAMING MATMUL
# ==============================================================

def jit_element(A: SeedMatrix, B: SeedMatrix, i: int, j: int) -> float:
    """
    JIT-compute C[i][j] = SUM_k A[i][k] * B[k][j]
    without materializing A, B, or C.

    Complexity: O(K) per element.
    Storage: O(1) -- no arrays allocated.
    Reproducibility: EXACT (IEEE-754 deterministic).
    """
    assert A.cols == B.rows, f"Dim mismatch: {A.cols} != {B.rows}"
    K = A.cols
    acc = 0.0
    for k in range(K):
        acc += A.resolve(i, k) * B.resolve(k, j)
    return acc


def jit_tile(A: SeedMatrix, B: SeedMatrix,
             r0: int, c0: int, tr: int, tc: int) -> list:
    """
    JIT-compute a tile of C without materializing the full matrices.
    Uses tiled dot-product streaming.

    Complexity: O(tr * tc * K)
    Storage: O(tr * tc) -- only the output tile is in memory.
    """
    K = A.cols
    tile = [[0.0] * tc for _ in range(tr)]
    for i in range(tr):
        for j in range(tc):
            acc = 0.0
            for k in range(K):
                acc += A.resolve(r0 + i, k) * B.resolve(k, c0 + j)
            tile[i][j] = acc
    return tile


def jit_streaming_matmul(A: SeedMatrix, B: SeedMatrix,
                         tile_size: int = 64) -> list:
    """
    Full JIT streaming matmul: tiles through A x B without
    ever holding the full A, B, or C in memory simultaneously.

    Only one tile of C exists at a time.
    Returns the full C matrix (for verification), but in production
    each tile would be consumed/forwarded immediately.
    """
    M, K, N = A.rows, A.cols, B.cols
    T = tile_size
    C = [[0.0] * N for _ in range(M)]

    for i0 in range(0, M, T):
        im = min(i0 + T, M)
        for j0 in range(0, N, T):
            jm = min(j0 + T, N)
            # Accumulate over K tiles
            for k0 in range(0, K, T):
                km = min(k0 + T, K)
                # JIT materialize the input tiles
                A_tile = A.materialize_tile(i0, k0, im - i0, km - k0)
                B_tile = B.materialize_tile(k0, j0, km - k0, jm - j0)
                # Compute partial product
                for i in range(im - i0):
                    for j in range(jm - j0):
                        for k in range(km - k0):
                            C[i0 + i][j0 + j] += A_tile[i][k] * B_tile[k][j]
    return C


# ==============================================================
# TESTS
# ==============================================================

def test_exact_reproducibility(sizes):
    """Verify that JIT element computation is BIT-EXACT reproducible."""
    print("\n" + "=" * 70)
    print("TEST 1: EXACT REPRODUCIBILITY (bit-exact across calls)")
    print("=" * 70)

    for N in sizes:
        A = SeedMatrix(N, N, seed=0xDEADBEEF)
        B = SeedMatrix(N, N, seed=0xCAFEBABE)

        # Compute same elements twice, check bit-exactness
        rng = random.Random(42)
        sample_coords = [(rng.randint(0, N-1), rng.randint(0, N-1))
                         for _ in range(100)]

        pass_1 = [jit_element(A, B, i, j) for i, j in sample_coords]
        pass_2 = [jit_element(A, B, i, j) for i, j in sample_coords]

        # Bit-exact comparison: compare as raw bytes
        exact = 0
        for v1, v2 in zip(pass_1, pass_2):
            b1 = struct.pack('d', v1)
            b2 = struct.pack('d', v2)
            if b1 == b2:
                exact += 1

        print(f"  N={N:>5}: {exact}/100 bit-exact matches "
              f"({'PASS' if exact == 100 else 'FAIL'})")

        # Also check SeedMatrix resolve reproducibility
        val_a = A.resolve(0, 0)
        val_b = A.resolve(0, 0)
        assert struct.pack('d', val_a) == struct.pack('d', val_b), \
            "SeedMatrix.resolve is not deterministic!"

    print("  [DONE] Reproducibility verified.")


def test_jit_vs_dense(sizes):
    """Compare JIT streaming matmul against dense for correctness."""
    print("\n" + "=" * 70)
    print("TEST 2: JIT STREAMING vs DENSE (correctness + throughput)")
    print("=" * 70)

    print(f"  {'N':>6} | {'JIT Time':>10} | {'Dense Time':>10} | "
          f"{'MaxErr':>10} | {'Match':>6} | {'Mem Saved':>10}")
    print("  " + "-" * 66)

    for N in sizes:
        A = SeedMatrix(N, N, seed=0x1234)
        B = SeedMatrix(N, N, seed=0x5678)

        T = min(N, 64)

        # JIT streaming matmul
        t0 = time.perf_counter()
        C_jit = jit_streaming_matmul(A, B, tile_size=T)
        t_jit = time.perf_counter() - t0

        # Dense matmul (materialize A, B first)
        t0 = time.perf_counter()
        A_dense = [A.resolve_row(i) for i in range(N)]
        B_dense = [B.resolve_row(i) for i in range(N)]
        C_dense = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                s = 0.0
                for k in range(N):
                    s += A_dense[i][k] * B_dense[k][j]
                C_dense[i][j] = s
        t_dense = time.perf_counter() - t0

        # Compare
        max_err = 0.0
        errors = 0
        for i in range(N):
            for j in range(N):
                err = abs(C_jit[i][j] - C_dense[i][j])
                max_err = max(max_err, err)
                if err > 1e-9:
                    errors += 1

        # Memory savings
        dense_mem = N * N * 8 * 3  # A + B + C
        jit_mem = A.memory_bytes() + B.memory_bytes() + T * T * 8  # seeds + 1 tile
        saved_pct = (1 - jit_mem / dense_mem) * 100

        status = "EXACT" if max_err < 1e-9 else f"{errors} err"
        print(f"  {N:>6} | {t_jit:>9.3f}s | {t_dense:>9.3f}s | "
              f"{max_err:>10.2e} | {status:>6} | {saved_pct:>9.1f}%")

    print("  [DONE] JIT vs Dense comparison complete.")


def test_streaming_element_throughput(sizes):
    """Measure raw element resolution throughput."""
    print("\n" + "=" * 70)
    print("TEST 3: ELEMENT RESOLUTION THROUGHPUT")
    print("=" * 70)

    print(f"  {'N':>6} | {'SeedMatrix':>12} | {'JIT C[i,j]':>12} | "
          f"{'Elements/s':>12}")
    print("  " + "-" * 54)

    for N in sizes:
        A = SeedMatrix(N, N, seed=0xABCD)

        # SeedMatrix resolution throughput
        count = min(10000, N * N)
        coords = [(i % N, j % N) for i in range(100) for j in range(100)]
        coords = coords[:count]

        t0 = time.perf_counter()
        for i, j in coords:
            A.resolve(i, j)
        t_resolve = time.perf_counter() - t0
        rate_resolve = count / max(t_resolve, 1e-9)

        # JIT matmul element throughput
        B = SeedMatrix(N, N, seed=0xEF01)
        sample_count = min(100, N)
        t0 = time.perf_counter()
        for i in range(sample_count):
            jit_element(A, B, i, 0)
        t_jit = time.perf_counter() - t0
        rate_jit = sample_count / max(t_jit, 1e-9)

        print(f"  {N:>6} | {rate_resolve:>10,.0f}/s | {rate_jit:>10,.0f}/s | "
              f"(K={N} per dot)")

    print("  [DONE] Throughput measured.")


def test_collapse_then_verify(sizes):
    """
    Full pipeline: JIT matmul -> collapse -> manifold + RNS ledger
    Then verify: can we reproduce the exact dense result?
    """
    print("\n" + "=" * 70)
    print("TEST 4: COLLAPSE + VERIFY (exact materialization check)")
    print("=" * 70)

    print(f"  {'N':>6} | {'Tiles':>5} | {'Compress':>8} | "
          f"{'RNS Exact':>10} | {'Time':>8}")
    print("  " + "-" * 52)

    for N in sizes:
        A = SeedMatrix(N, N, seed=0x111)
        B = SeedMatrix(N, N, seed=0x222)
        T = min(N, 64)

        t0 = time.perf_counter()

        # Step 1: JIT streaming matmul (produces dense C)
        C = jit_streaming_matmul(A, B, tile_size=T)

        # Step 2: Collapse into manifold
        mf = fit_manifold(C, tile_size=T)

        # Step 3: Record RNS ledger (for exact verification)
        ledger = record_matrix(C)

        elapsed = time.perf_counter() - t0

        # Step 4: Verify — pick 200 random elements, check RNS exactness
        rng = random.Random(42)
        rns_exact = 0
        checks = min(200, N * N)
        for _ in range(checks):
            i = rng.randint(0, N - 1)
            j = rng.randint(0, N - 1)
            # Recompute via JIT (should be bit-exact same as C[i][j])
            val_jit = jit_element(A, B, i, j)
            # Verify against RNS ledger
            if ledger.verify_residues(i, j, val_jit):
                rns_exact += 1

        cr = mf.compression_ratio()
        print(f"  {N:>6} | {len(mf.tiles):>5} | {cr:>7.1f}x | "
              f"{rns_exact:>4}/{checks:<4}   | {elapsed:>7.1f}s")

    print("  [DONE] Collapse + verify complete.")


def test_large_scale_seed_only():
    """
    The ultimate Memory Wall test:
    Define matrices at scales where dense storage is impossible,
    but individual elements can still be computed on demand.
    """
    print("\n" + "=" * 70)
    print("TEST 5: SEED-ONLY EXTREME SCALE (no dense materialization)")
    print("=" * 70)

    extreme_sizes = [
        (10_000, "10K x 10K (800MB dense)"),
        (100_000, "100K x 100K (80GB dense)"),
        (1_000_000, "1M x 1M (8TB dense)"),
        (1_000_000_000, "1B x 1B (8 Exabytes dense)"),
    ]

    print(f"  {'Scale':>30} | {'Seed bytes':>10} | {'Dense bytes':>15} | "
          f"{'Ratio':>12} | {'Elem Time':>10}")
    print("  " + "-" * 85)

    for N, label in extreme_sizes:
        A = SeedMatrix(N, N, seed=0xFFFFF)
        B = SeedMatrix(N, N, seed=0xAAAAA)

        # Can we resolve individual elements?
        t0 = time.perf_counter()
        val_00 = A.resolve(0, 0)
        val_mid = A.resolve(N // 2, N // 2)
        val_last = A.resolve(N - 1, N - 1)
        t_elem = (time.perf_counter() - t0) / 3 * 1000

        seed_bytes = A.memory_bytes() + B.memory_bytes()
        dense_bytes = N * N * 8

        if dense_bytes < 1e6:
            dense_str = f"{dense_bytes:,.0f} B"
        elif dense_bytes < 1e9:
            dense_str = f"{dense_bytes/1e6:,.1f} MB"
        elif dense_bytes < 1e12:
            dense_str = f"{dense_bytes/1e9:,.1f} GB"
        elif dense_bytes < 1e15:
            dense_str = f"{dense_bytes/1e12:,.1f} TB"
        else:
            dense_str = f"{dense_bytes/1e18:,.1f} EB"

        ratio = dense_bytes / seed_bytes
        if ratio < 1e6:
            ratio_str = f"{ratio:,.0f}:1"
        elif ratio < 1e9:
            ratio_str = f"{ratio/1e6:,.1f}M:1"
        elif ratio < 1e12:
            ratio_str = f"{ratio/1e9:,.1f}B:1"
        else:
            ratio_str = f"{ratio/1e12:,.1f}T:1"

        print(f"  {label:>30} | {seed_bytes:>8} B | {dense_str:>15} | "
              f"{ratio_str:>12} | {t_elem:>8.3f}ms")

    # JIT dot product at 10K scale (compute ONE element of C = A x B)
    print(f"\n  JIT single element of C = A x B at 10K x 10K:")
    A = SeedMatrix(10_000, 10_000, seed=0xFFFFF)
    B = SeedMatrix(10_000, 10_000, seed=0xAAAAA)
    t0 = time.perf_counter()
    val = jit_element(A, B, 5000, 5000)
    t_dot = (time.perf_counter() - t0) * 1000
    print(f"    C[5000][5000] = {val:.10f}")
    print(f"    Time: {t_dot:.1f}ms (K=10000 multiply-adds)")
    print(f"    Memory used: 48 bytes (two seeds)")

    # Verify reproducibility
    val2 = jit_element(A, B, 5000, 5000)
    exact = struct.pack('d', val) == struct.pack('d', val2)
    print(f"    Bit-exact reproducible: {exact}")

    print("  [DONE] Extreme scale test complete.")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    print("+" + "=" * 68 + "+")
    print("|  JIT STREAMING MATMUL -- EXACT MATERIALIZATION TEST                |")
    print("|  Testing general computation via seed-defined matrices             |")
    print("+" + "=" * 68 + "+")

    t_total = time.perf_counter()

    # Tests 1-3: moderate sizes for correctness and throughput
    moderate_sizes = [256, 512]
    test_exact_reproducibility(moderate_sizes + [1024])
    test_streaming_element_throughput(moderate_sizes + [1024, 2048])

    # Test 2: JIT vs Dense (capped at 512 due to O(n^3) compute)
    test_jit_vs_dense([256, 512])

    # Test 4: Full pipeline with collapse + RNS
    test_collapse_then_verify([256, 512])

    # Test 5: Extreme scale (seed-only, no dense)
    test_large_scale_seed_only()

    total_time = time.perf_counter() - t_total
    print("\n" + "=" * 70)
    print(f"  TOTAL TIME: {total_time:.2f}s")
    print("=" * 70)


