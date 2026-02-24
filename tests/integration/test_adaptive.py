"""
Adaptive Classification System — Verification Suite.

Tests:
  1. Cost gate: small matrices bypass classification
  2. Feature vector: correct extraction from various matrix types
  3. Strategy cache: learning and promotion
  4. QMatrix routing: large matrices delegate to tiled engine
  5. Accuracy: all engines produce correct results vs naive
"""
import sys, os, time
import numpy as np

# Ensure the SDK root is on the path (parent of matrix_v_sdk)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from matrix_v_sdk.vl.substrate.matrix import (
    MatrixOmega, MatrixFeatureVector, StrategyCache,
    StrategyRecord, SymbolicDescriptor, SHUNT_THRESHOLD
)

def rel_error(gt, pred):
    gt, pred = np.asarray(gt, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    return float(np.linalg.norm(gt - pred) / (np.linalg.norm(gt) + 1e-15))


# ── 1. COST GATE ──────────────────────────────────────────
def test_cost_gate():
    omega = MatrixOmega()
    print("=== COST GATE ===")
    small_sizes = [4, 8, 16, 32, 64]
    for n in small_sizes:
        A = np.random.rand(n, n).tolist()
        B = np.random.rand(n, n).tolist()
        strategy = omega.auto_select_strategy(A, B)
        flops = n * n * n
        _ = "dense" if flops < SHUNT_THRESHOLD else "NOT dense"  # expected
        status = "OK" if (flops < SHUNT_THRESHOLD) == (strategy == "dense") else "FAIL"
        print(f"  {status} n={n:>4}, FLOPs={flops:>10}, strategy={strategy:<16} (threshold={SHUNT_THRESHOLD})")


# ── 2. FEATURE VECTOR ──────────────────────────────────────
def test_feature_vector():
    print("\n=== FEATURE VECTOR EXTRACTION ===")

    # Dense random
    A = np.random.rand(64, 64).tolist()
    B = np.random.rand(64, 64).tolist()
    fv = MatrixFeatureVector.from_matrices(A, B)
    print(f"  Random 64x64:   sparsity={fv.sparsity:.3f}, var={fv.row_variance:.3f}, tile={fv.tile_periodicity:.3f}")

    # Sparse
    S = np.zeros((64, 64))
    S[0, 0] = 1.0; S[32, 32] = 1.0
    fv2 = MatrixFeatureVector.from_matrices(S.tolist(), B)
    print(f"  Sparse 64x64:   sparsity={fv2.sparsity:.3f}, var={fv2.row_variance:.3f}, tile={fv2.tile_periodicity:.3f}")

    # Tiled (repeating pattern)
    tile = [1.0, 2.0, 3.0, 4.0]
    T = [tile * 4 for _ in range(16)]
    B16 = np.random.rand(16, 16).tolist()
    fv3 = MatrixFeatureVector.from_matrices(T, B16)
    print(f"  Tiled 16x16:    sparsity={fv3.sparsity:.3f}, var={fv3.row_variance:.3f}, tile={fv3.tile_periodicity:.3f}")

    # Rectangular bottleneck
    A_rect = np.random.rand(10, 1000).tolist()
    B_rect = np.random.rand(1000, 10).tolist()
    fv4 = MatrixFeatureVector.from_matrices(A_rect, B_rect)
    print(f"  Rect 10x1000:   bottleneck={fv4.is_rectangular_bottleneck}, shape_class={fv4.shape_class()}")


# ── 3. STRATEGY CACHE LEARNING ─────────────────────────────
def test_strategy_cache():
    print("\n=== STRATEGY CACHE LEARNING ===")
    cache = StrategyCache()

    # Simulate repeated observations
    shape = "128x128x128"
    for i in range(10):
        cache.observe(shape, "qmatrix", error=0.0, time_ms=15.0 + np.random.randn() * 0.5)
        cache.observe(shape, "dense", error=0.0, time_ms=50.0 + np.random.randn() * 1.0)
        cache.observe(shape, "spectral", error=1e-4, time_ms=30.0 + np.random.randn() * 0.5)

    promoted = cache.recall(shape)
    print(f"  After 10 observations, promoted engine: {promoted}")

    # Check utility scores
    for eng in ["qmatrix", "dense", "spectral"]:
        rec = cache._cache.get(f"{shape}:{eng}")
        if rec:
            print(f"    {eng:>12}: utility={rec.spectral_utility():.3f}, avg_ms={rec.avg_time_ms:.1f}, confident={rec.is_confident()}")


# ── 4. QMATRIX ROUTING ────────────────────────────────────
def test_qmatrix_routing():
    print("\n=== QMATRIX ROUTING ===")
    omega = MatrixOmega()
    sizes = [128, 256, 512, 1024]
    for n in sizes:
        A = np.random.rand(n, n).tolist()[:min(n, 8)]  # fake large matrix (just strategy check)
        A_full = [[0.0] * n for _ in range(n)]
        for i in range(min(n, 8)):
            A_full[i] = A[i] + [0.0] * (n - len(A[i]))
        B = [[0.0] * n for _ in range(n)]
        strategy = omega.auto_select_strategy(A_full, B)
        expected = "qmatrix" if n > 512 else "N/A"
        print(f"  n={n:>5}: strategy={strategy:<16} (expected QMatrix for n>512)")


# ── 5. ACCURACY ────────────────────────────────────────────
def test_accuracy():
    print("\n=== ACCURACY (All Engines via Adaptive Dispatch) ===")
    omega = MatrixOmega()
    header = f"{'n':>5} | {'strategy':>16} | {'error':>12} | {'time_ms':>10}"
    print(header)
    print("-" * len(header))

    sizes = [8, 16, 32, 64, 128]
    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        GT = A @ B

        t0 = time.perf_counter()
        result = omega.compute_product(A.tolist(), B.tolist())
        t = time.perf_counter() - t0
        err = rel_error(GT, result)
        strategy = omega.auto_select_strategy(A.tolist(), B.tolist())
        print(f"{n:>5} | {strategy:>16} | {err:>12.2e} | {t*1e3:>10.3f}")


# ── 6. CACHE PERSISTENCE (warm-up test) ───────────────────
def test_warm_dispatch():
    print("\n=== WARM DISPATCH (Cache Learning Effect) ===")
    omega = MatrixOmega()
    n = 128
    A = np.random.rand(n, n).tolist()
    B = np.random.rand(n, n).tolist()

    # Cold pass
    t0 = time.perf_counter()
    omega.compute_product(A, B)
    cold_ms = (time.perf_counter() - t0) * 1000

    # Warm passes
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        omega.compute_product(A, B)
        times.append((time.perf_counter() - t0) * 1000)

    fv = MatrixFeatureVector.from_matrices(A, B)
    promoted = omega.strategy_cache.recall(fv.shape_class())
    print(f"  Cold pass: {cold_ms:.3f}ms")
    print(f"  Warm avg:  {sum(times)/len(times):.3f}ms")
    print(f"  Promoted:  {promoted}")


# ── 7. CONFIDENCE GATE ─────────────────────────────────────
def test_confidence_gate():
    print("\n=== CONFIDENCE GATE (vGPU z*sigma/sqrt(n) < eps*|mu|) ===")
    rec = StrategyRecord(engine_name="test")

    # Feed stable observations
    for i in range(20):
        rec.record(error=0.0, time_ms=10.0 + 0.01 * np.random.randn())
        if rec.is_confident():
            print(f"  Promoted after {i+1} observations (avg={rec.avg_time_ms:.3f}ms)")
            break
    else:
        print(f"  Not promoted after 20 observations")

    # Feed noisy observations
    rec2 = StrategyRecord(engine_name="noisy")
    for i in range(20):
        rec2.record(error=0.1, time_ms=10.0 + 5.0 * np.random.randn())
    print(f"  Noisy engine confident: {rec2.is_confident()} (expected False)")


if __name__ == "__main__":
    test_cost_gate()
    test_feature_vector()
    test_strategy_cache()
    test_qmatrix_routing()
    test_accuracy()
    test_warm_dispatch()
    test_confidence_gate()


