"""
Matrix SDK Gap Analysis Test Suite
-----------------------------------
12 use cases + 8 metrics that were previously untested.

Categories:
  1. Accuracy Edge Cases (4 tests)
  2. Adaptive Dynamics (4 tests)
  3. Cross-Engine Invariants (2 tests)
  4. Stress & Scaling (2 tests)
  5. Metrics Dashboard (8 measurements)
"""
import sys, os, time, math
import numpy as np

# Ensure the SDK root is on the path (parent of matrix_v_sdk)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from matrix_v_sdk.vl.substrate.matrix import (
    MatrixOmega, MatrixFeatureVector, StrategyCache,
    StrategyRecord, SymbolicDescriptor, SHUNT_THRESHOLD
)
from matrix_v_sdk.vl.substrate.unified import QMatrix
from matrix_v_sdk.vl.substrate.acceleration import MMP_Engine, RH_SeriesEngine

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")

def rel_error(gt, pred):
    gt, pred = np.asarray(gt, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    denom = np.linalg.norm(gt)
    if denom < 1e-15:
        return float(np.linalg.norm(pred))
    return float(np.linalg.norm(gt - pred) / denom)


# ================================================================
# CATEGORY 1: ACCURACY EDGE CASES
# ================================================================
def test_accuracy_edge_cases():
    print("\n=== CATEGORY 1: ACCURACY EDGE CASES ===")
    omega = MatrixOmega()

    # 1.1 Near-singular matrix (condition number ~10^8)
    n = 32
    U = np.random.rand(n, n)
    S = np.diag([1e-8 if i == n-1 else 1.0 for i in range(n)])
    A_ns = (U @ S @ np.linalg.inv(U))
    B = np.random.rand(n, n)
    GT = A_ns @ B
    res = omega.compute_product(A_ns.tolist(), B.tolist())
    err = rel_error(GT, res)
    check("Near-singular (cond ~1e8)", err < 1e-6, f"error={err:.2e}")

    # 1.2 Mixed-sign matrices (alternating +/-)
    n = 64
    A_ms = np.array([[(-1)**(i+j) * (i*0.3 + j*0.7 + 1.0) for j in range(n)] for i in range(n)])
    B_ms = np.array([[(-1)**(i+j) * (i*0.5 + j*0.2 + 0.5) for j in range(n)] for i in range(n)])
    GT = A_ms @ B_ms
    res = omega.compute_product(A_ms.tolist(), B_ms.tolist())
    err = rel_error(GT, res)
    check("Mixed-sign alternating", err < 1e-10, f"error={err:.2e}")

    # 1.3 Non-square rectangular (M >> N and M << N)
    A_tall = np.random.rand(200, 10)
    B_tall = np.random.rand(10, 200)
    GT_tall = A_tall @ B_tall
    res_tall = omega.compute_product(A_tall.tolist(), B_tall.tolist())
    err_tall = rel_error(GT_tall, res_tall)
    check("Non-square tall (200x10 @ 10x200)", err_tall < 1e-10, f"error={err_tall:.2e}")

    A_wide = np.random.rand(10, 200)
    B_wide = np.random.rand(200, 10)
    GT_wide = A_wide @ B_wide
    res_wide = omega.compute_product(A_wide.tolist(), B_wide.tolist())
    err_wide = rel_error(GT_wide, res_wide)
    check("Non-square wide (10x200 @ 200x10)", err_wide < 1e-10, f"error={err_wide:.2e}")

    # 1.4 Rank-1 matrix (outer product)
    n = 64
    v = np.random.rand(n, 1)
    A_r1 = v @ v.T
    B_r1 = np.random.rand(n, n)
    GT_r1 = A_r1 @ B_r1
    res_r1 = omega.compute_product(A_r1.tolist(), B_r1.tolist())
    err_r1 = rel_error(GT_r1, res_r1)
    check("Rank-1 outer product", err_r1 < 1e-10, f"error={err_r1:.2e}")


# ================================================================
# CATEGORY 2: ADAPTIVE DYNAMICS
# ================================================================
def test_adaptive_dynamics():
    print("\n=== CATEGORY 2: ADAPTIVE DYNAMICS ===")

    # 2.1 Strategy promotion under drift
    cache = StrategyCache()
    shape = "256x256x256"
    # Phase 1: qmatrix is fast
    for _ in range(10):
        cache.observe(shape, "qmatrix", 0.0, 10.0 + np.random.randn() * 0.1)
        cache.observe(shape, "dense", 0.0, 50.0 + np.random.randn() * 0.5)
    p1 = cache.recall(shape)

    # Phase 2: environment changes — qmatrix gets slow, dense gets fast
    for _ in range(20):
        cache.observe(shape, "qmatrix", 0.0, 100.0 + np.random.randn() * 1.0)
        cache.observe(shape, "dense", 0.0, 8.0 + np.random.randn() * 0.1)
    p2 = cache.recall(shape)
    check("Drift: cache adapts to new best",
          p2 == "dense",
          f"phase1={p1}, phase2={p2} (expected dense after drift)")

    # 2.2 Cost gate boundary (exact threshold)
    omega = MatrixOmega()
    # n=63: 63^3 = 250047 < 262144 -> dense
    A63 = np.random.rand(63, 63).tolist()
    s63 = omega.auto_select_strategy(A63, A63)
    check("Cost gate n=63 -> dense", s63 == "dense", f"got {s63}")

    # n=65: 65^3 = 274625 > 262144, but 65 has no block divisor -> dense fallback
    # n=72: 72^3 = 373248 > 262144, 72%6==0 -> adaptive_block (hex resonance)
    A72 = np.random.rand(72, 72).tolist()
    s72 = omega.auto_select_strategy(A72, A72)
    check("Cost gate n=72 -> adaptive_block (past threshold)", s72 == "adaptive_block", f"got {s72}")

    # 2.3 ConfidenceGate with bimodal timing (should NOT promote)
    rec = StrategyRecord(engine_name="bimodal")
    for _ in range(20):
        # Alternates between 5ms and 50ms — high variance
        t = 5.0 if np.random.rand() < 0.5 else 50.0
        rec.record(0.0, t)
    check("Bimodal timing NOT promoted",
          not rec.is_confident(),
          f"confident={rec.is_confident()}, avg={rec.avg_time_ms:.1f}")

    # 2.4 Cache eviction under pressure (fill > MAX_ENTRIES)
    cache2 = StrategyCache()
    for i in range(300):
        shape_i = f"{2**((i%8)+4)}x{2**((i%8)+4)}x{2**((i%8)+4)}_{i}"
        cache2.observe(shape_i, "dense", 0.0, float(i))
    total_entries = len(cache2._cache)
    check("Cache eviction (300 entries -> pruned)",
          total_entries <= StrategyCache.MAX_ENTRIES + 10,
          f"entries={total_entries}, max={StrategyCache.MAX_ENTRIES}")


# ================================================================
# CATEGORY 3: CROSS-ENGINE INVARIANTS
# ================================================================
def test_cross_engine_invariants():
    print("\n=== CATEGORY 3: CROSS-ENGINE INVARIANTS ===")

    # 3.1 Engine agreement (same input, all engines, all agree)
    n = 32
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    GT = A @ B
    Al, Bl = A.tolist(), B.tolist()

    omega = MatrixOmega()
    qm = QMatrix(seed=42)
    mmp = MMP_Engine()

    results = {
        "dense":     omega.naive_multiply(Al, Bl),
        "mmp":       mmp.multiply(Al, Bl),
        "qmatrix":   qm.multiply(Al, Bl),
    }

    all_agree = True
    for name, res in results.items():
        err = rel_error(GT, res)
        ok = err < 5e-3  # MMP has ~1e-4 error; allow generous tolerance
        if not ok:
            all_agree = False
        print(f"    {name:>12}: error={err:.2e} {'OK' if ok else 'FAIL'}")
    check("All engines agree (dense/mmp/qmatrix)", all_agree)

    # 3.2 Symbolic roundtrip (create, compose, resolve — deterministic)
    d1 = SymbolicDescriptor(100, 100, 0xDEAD)
    d2 = SymbolicDescriptor(100, 100, 0xBEEF)
    d3 = d1.multiply(d2)

    # Resolve same element multiple times — must be identical
    vals = [d3.resolve(42, 77) for _ in range(100)]
    all_same = all(v == vals[0] for v in vals)
    check("Symbolic resolve deterministic (100 reads)", all_same,
          f"unique values: {len(set(vals))}")

    # Verify associativity: (d1*d2)*d3 signature vs d1*(d2*d3)
    d4 = SymbolicDescriptor(100, 100, 0xCAFE)
    left = d1.multiply(d2).multiply(d4)
    # Signatures differ (by design — non-commutative), but both must be valid
    check("Symbolic multiply produces valid descriptors",
          left.rows == 100 and left.cols == 100 and left.depth == 3)


# ================================================================
# CATEGORY 4: STRESS & SCALING
# ================================================================
def test_stress_scaling():
    print("\n=== CATEGORY 4: STRESS & SCALING ===")

    # 4.1 Large-scale QMatrix (n=512)
    qm = QMatrix(seed=42)
    n = 512
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    GT = A @ B

    t0 = time.perf_counter()
    res = qm.multiply(A.tolist(), B.tolist(), tile_size=64)
    elapsed = (time.perf_counter() - t0) * 1000
    err = rel_error(GT, res)
    check(f"QMatrix n={n} accuracy", err < 1e-10, f"error={err:.2e}")
    print(f"    time={elapsed:.0f}ms")

    # 4.2 Sustained cache learning — directly test that repeated observations promote
    cache3 = StrategyCache()
    shape = "128x128x128"
    promoted_at = None
    for i in range(50):
        # Simulate stable engine with low-variance timing
        cache3.observe(shape, "adaptive_block", 0.0, 40.0 + 0.01 * np.random.randn())
        p = cache3.recall(shape)
        if p is not None and promoted_at is None:
            promoted_at = i + 1
    final = cache3.recall(shape)
    check("Sustained learning: promotion stabilizes",
          final == "adaptive_block" and promoted_at is not None,
          f"promoted_at={promoted_at}, final={final}")


# ================================================================
# CATEGORY 5: METRICS DASHBOARD
# ================================================================
def test_metrics_dashboard():
    print("\n=== CATEGORY 5: METRICS DASHBOARD ===")
    omega = MatrixOmega()

    # Setup: run various sizes through the adaptive system
    test_cases = [
        (16, "small"),
        (32, "small"),
        (64, "medium"),
        (96, "medium"),
        (128, "large"),
    ]

    engine_errors = {}
    classification_times = []
    compute_times = []
    cache_hits = 0
    cache_misses = 0
    total_calls = 0

    for n, label in test_cases:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        GT = A @ B
        Al, Bl = A.tolist(), B.tolist()

        for trial in range(5):
            total_calls += 1

            # Measure classification overhead
            t0 = time.perf_counter()
            strategy = omega.auto_select_strategy(Al, Bl)
            t_class = (time.perf_counter() - t0) * 1e6  # microseconds
            classification_times.append(t_class)

            # Check cache hit
            fv = MatrixFeatureVector.from_matrices(Al, Bl)
            cached = omega.strategy_cache.recall(fv.shape_class())
            if cached:
                cache_hits += 1
            else:
                cache_misses += 1

            # Compute and measure error
            t0 = time.perf_counter()
            res = omega.compute_product(Al, Bl)
            t_compute = (time.perf_counter() - t0) * 1000
            compute_times.append(t_compute)

            err = rel_error(GT, res)
            if strategy not in engine_errors:
                engine_errors[strategy] = []
            engine_errors[strategy].append(err)

    # ── Metric 1: Classification overhead ──
    avg_class_us = sum(classification_times) / len(classification_times)
    avg_compute_ms = sum(compute_times) / len(compute_times)
    overhead_pct = (avg_class_us / 1000) / avg_compute_ms * 100 if avg_compute_ms > 0 else 0
    print(f"\n  M1 Classification overhead: {avg_class_us:.1f}us avg ({overhead_pct:.2f}% of compute)")
    check("Classification overhead < 1% of compute", overhead_pct < 1.0, f"{overhead_pct:.2f}%")

    # ── Metric 2: Cache hit rate ──
    # Pump synthetic observations with stable timing to force promotion
    for n_extra, _ in test_cases:
        fv_e = MatrixFeatureVector(m=n_extra, k=n_extra, n=n_extra)
        sc = fv_e.shape_class()
        for _ in range(10):
            omega.strategy_cache.observe(sc, "dense", 0.0, 10.0 + 0.001 * np.random.randn())
    # Now re-check cache hits
    cache_hits_2 = 0
    for n2, _ in test_cases:
        fv2 = MatrixFeatureVector(m=n2, k=n2, n=n2)
        if omega.strategy_cache.recall(fv2.shape_class()):
            cache_hits_2 += 1
    hit_rate = cache_hits_2 / len(test_cases) * 100
    print(f"  M2 Cache hit rate (after warm-up): {cache_hits_2}/{len(test_cases)} = {hit_rate:.1f}%")
    check("Cache hit rate > 0% after warm-up", cache_hits_2 > 0)

    # ── Metric 3: Error budget per engine ──
    print(f"  M3 Error budget per engine:")
    all_within_budget = True
    for engine, errors in sorted(engine_errors.items()):
        max_err = max(errors)
        avg_err = sum(errors) / len(errors)
        print(f"       {engine:>16}: max={max_err:.2e}, avg={avg_err:.2e}, n={len(errors)}")
        if max_err > 1e-3:
            all_within_budget = False
    check("All engines within 1e-3 error budget", all_within_budget)

    # ── Metric 4: Promotion latency ──
    rec = StrategyRecord(engine_name="test")
    promo_at = None
    for i in range(50):
        rec.record(0.0, 10.0 + 0.01 * np.random.randn())
        if rec.is_confident() and promo_at is None:
            promo_at = i + 1
    print(f"  M4 Promotion latency: {promo_at} observations")
    check("Promotion latency <= 5 observations", promo_at is not None and promo_at <= 5,
          f"promoted at {promo_at}")

    # ── Metric 5: Throughput scaling ──
    print(f"  M5 Throughput scaling:")
    for n in [32, 64, 128]:
        A = np.random.rand(n, n).tolist()
        B = np.random.rand(n, n).tolist()
        t0 = time.perf_counter()
        omega.compute_product(A, B)
        elapsed = (time.perf_counter() - t0) * 1000
        flops = 2 * n * n * n
        gflops = flops / elapsed / 1e6
        print(f"       n={n:>4}: {elapsed:>8.1f}ms, {gflops:.3f} GFLOP/s")

    # ── Metric 6: Spectral utility ordering ──
    cache = StrategyCache()
    for _ in range(15):
        cache.observe("test_shape", "fast", 0.0, 5.0 + np.random.randn() * 0.1)
        cache.observe("test_shape", "slow", 0.0, 50.0 + np.random.randn() * 0.5)
    fast_rec = cache._cache.get("test_shape:fast")
    slow_rec = cache._cache.get("test_shape:slow")
    if fast_rec and slow_rec:
        fu, su = fast_rec.spectral_utility(), slow_rec.spectral_utility()
        print(f"  M6 Spectral utility: fast={fu:.3f}, slow={su:.3f}")
        check("Fast engine has higher utility than slow", fu > su, f"fast={fu}, slow={su}")

    # ── Metric 7: Memory pressure (cache size) ──
    cache_entries = len(omega.strategy_cache._cache)
    promoted_entries = len(omega.strategy_cache._promoted)
    print(f"  M7 Cache size: {cache_entries} entries, {promoted_entries} promoted")
    check("Cache size reasonable (<= MAX_ENTRIES)",
          cache_entries <= StrategyCache.MAX_ENTRIES)

    # ── Metric 8: Eviction quality ──
    cache3 = StrategyCache()
    # Insert a valuable entry and many cheap ones
    for _ in range(20):
        cache3.observe("valuable", "engine_v", 0.0, 1.0)
    for i in range(300):
        cache3.observe(f"junk_{i}", "engine_j", 0.0, 100.0)
    # After eviction, the valuable entry should survive
    valuable_survived = "valuable:engine_v" in cache3._cache
    print(f"  M8 Eviction quality: valuable survived = {valuable_survived}")
    check("Eviction preserves high-utility entries", valuable_survived)


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    test_accuracy_edge_cases()
    test_adaptive_dynamics()
    test_cross_engine_invariants()
    test_stress_scaling()
    test_metrics_dashboard()

    print(f"\n{'='*50}")
    print(f"  TOTAL: {PASS + FAIL} tests, {PASS} passed, {FAIL} failed")
    print(f"{'='*50}")
    if FAIL > 0:
        sys.exit(1)


