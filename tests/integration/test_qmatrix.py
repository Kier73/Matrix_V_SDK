"""
QMatrix Verification Suite
Tests: tiled matmul accuracy, symbolic path, quantum rank estimation.
"""
import sys, os, time # Ensure the SDK root is on the path (parent of matrix_v_sdk)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from matrix_v_sdk.vl.substrate.matrix import MatrixOmega, SymbolicDescriptor
from matrix_v_sdk.vl.substrate.unified import QMatrix, estimate_tile_rank

def rel_error(gt, pred):
    gt, pred = np.asarray(gt, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    return float(np.linalg.norm(gt - pred) / (np.linalg.norm(gt) + 1e-15))

# ── 1. TILED MATMUL ACCURACY ────────────────────────────
def test_tiled_accuracy():
    qm = QMatrix(seed=42)
    sizes = [16, 32, 64, 128, 256]
    tile_sizes = [16, 32, 64]

    header = f"{'n':>5} | {'tile':>5} | {'error':>12} | {'time_ms':>10}"
    print("=== TILED MATMUL ACCURACY ===")
    print(header)
    print("-" * len(header))

    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        GT = A @ B
        Al, Bl = A.tolist(), B.tolist()

        for T in tile_sizes:
            if T > n:
                continue
            t0 = time.perf_counter()
            result = qm.multiply(Al, Bl, tile_size=T)
            t = time.perf_counter() - t0
            err = rel_error(GT, result)
            print(f"{n:>5} | {T:>5} | {err:>12.2e} | {t*1e3:>10.3f}")

# ── 2. SYMBOLIC COMPOSITION ─────────────────────────────
def test_symbolic():
    qm = QMatrix(seed=42)
    print("\n=== SYMBOLIC COMPOSITION ===")
    dims = [10**3, 10**6, 10**9, 10**12]
    header = f"{'n':>15} | {'time_us':>10} | {'sig':>18}"
    print(header)
    print("-" * len(header))

    for n in dims:
        a = SymbolicDescriptor(n, n, 0xAAAA)
        b = SymbolicDescriptor(n, n, 0xBBBB)
        t0 = time.perf_counter()
        c = qm.symbolic_multiply(a, b)
        t = time.perf_counter() - t0
        print(f"{n:>15,} | {t*1e6:>10.2f} | {c.signature:>18x}")

# ── 3. QUANTUM RANK ESTIMATION ──────────────────────────
def test_quantum_rank():
    print("\n=== QUANTUM RANK ESTIMATION ===")
    header = f"{'matrix_type':>20} | {'n':>5} | {'rank_ratio':>12} | {'expected':>10}"
    print(header)
    print("-" * len(header))

    # Low-rank: identity
    for n in [16, 32, 64]:
        I = np.eye(n).tolist()
        r = estimate_tile_rank(I)
        print(f"{'identity':>20} | {n:>5} | {r:>12.4f} | {'low':>10}")

    # Low-rank: rank-1
    for n in [16, 32, 64]:
        v = np.random.rand(n, 1)
        R1 = (v @ v.T).tolist()
        r = estimate_tile_rank(R1)
        print(f"{'rank-1':>20} | {n:>5} | {r:>12.4f} | {'low':>10}")

    # Full-rank: random
    for n in [16, 32, 64]:
        M = np.random.rand(n, n).tolist()
        r = estimate_tile_rank(M)
        print(f"{'random (full-rank)':>20} | {n:>5} | {r:>12.4f} | {'high':>10}")

    # Full-rank: scaled random
    for n in [16, 32, 64]:
        M = (np.random.rand(n, n) * 100).tolist()
        r = estimate_tile_rank(M)
        print(f"{'random*100':>20} | {n:>5} | {r:>12.4f} | {'high':>10}")

# ── 4. THROUGHPUT ────────────────────────────────────────
def test_throughput():
    qm = QMatrix(seed=42)
    omega = MatrixOmega()
    print("\n=== QMATRIX vs OMEGA THROUGHPUT ===")
    sizes = [64, 128, 256]
    header = f"{'n':>5} | {'QMatrix_ms':>12} | {'Omega_ms':>12} | {'ratio':>8}"
    print(header)
    print("-" * len(header))

    for n in sizes:
        A = np.random.rand(n, n).tolist()
        B = np.random.rand(n, n).tolist()

        t0 = time.perf_counter()
        qm.multiply(A, B)
        t_qm = time.perf_counter() - t0

        t0 = time.perf_counter()
        omega.compute_product(A, B)
        t_om = time.perf_counter() - t0

        ratio = t_om / t_qm if t_qm > 0 else 0
        print(f"{n:>5} | {t_qm*1e3:>12.3f} | {t_om*1e3:>12.3f} | {ratio:>8.2f}x")

if __name__ == "__main__":
    test_tiled_accuracy()
    test_symbolic()
    test_quantum_rank()
    test_throughput()


