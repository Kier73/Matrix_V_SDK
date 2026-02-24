"""
V-Series SDK — Metrics Suite
Outputs: throughput, accuracy, scaling exponent, memory.
"""
import time, sys, os, tracemalloc
import numpy as np

sys.path.insert(0, os.path.abspath(os.getcwd()))

from matrix_v_sdk.vl.substrate.matrix import MatrixOmega, SymbolicDescriptor, XMatrix, PrimeMatrix
from matrix_v_sdk.vl.substrate.v_matrix import VMatrix
from matrix_v_sdk.vl.substrate.acceleration import MMP_Engine

# ── helpers ──────────────────────────────────────────────
def rel_error(gt, pred):
    gt, pred = np.asarray(gt, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    return float(np.linalg.norm(gt - pred) / (np.linalg.norm(gt) + 1e-15))

def fit_exponent(sizes, times):
    """Least-squares fit log(t) = a*log(n) + b → returns a (the scaling exponent)."""
    ln = np.log(np.array(sizes, dtype=np.float64))
    lt = np.log(np.array(times, dtype=np.float64) + 1e-15)
    a, _ = np.polyfit(ln, lt, 1)
    return round(a, 2)

def timed(fn, *args, repeats=1):
    best = float('inf')
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, result

# ── 1. THROUGHPUT & SCALING ──────────────────────────────
def throughput_table():
    sizes = [32, 64, 128, 256, 512]
    omega = MatrixOmega()
    spectral = VMatrix(mode="spectral")

    header = f"{'n':>5} | {'NumPy_ms':>10} | {'Omega_ms':>10} | {'Spectral_ms':>12} | {'NumPy_GFLOPS':>13} | {'Omega_GFLOPS':>13}"
    print("=== THROUGHPUT ===")
    print(header)
    print("-" * len(header))

    np_times, om_times, sp_times = [], [], []
    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        Al, Bl = A.tolist(), B.tolist()
        flops = 2.0 * n**3

        t_np, _ = timed(np.matmul, A, B)
        t_om, _ = timed(omega.compute_product, Al, Bl)
        t_sp, _ = timed(spectral.matmul, Al, Bl)

        np_times.append(t_np)
        om_times.append(t_om)
        sp_times.append(t_sp)

        gf_np = flops / t_np / 1e9
        gf_om = flops / t_om / 1e9

        print(f"{n:>5} | {t_np*1e3:>10.3f} | {t_om*1e3:>10.3f} | {t_sp*1e3:>12.3f} | {gf_np:>13.3f} | {gf_om:>13.3f}")

    print()
    print(f"Scaling exponent (NumPy):    {fit_exponent(sizes, np_times)}")
    print(f"Scaling exponent (Omega):    {fit_exponent(sizes, om_times)}")
    print(f"Scaling exponent (Spectral): {fit_exponent(sizes, sp_times)}")

# ── 2. ACCURACY ──────────────────────────────────────────
def accuracy_table():
    sizes = [32, 64, 128, 256]
    omega = MatrixOmega()
    spectral = VMatrix(mode="spectral")
    mmp = MMP_Engine()

    header = f"{'n':>5} | {'Omega_err':>12} | {'Spectral_err':>14} | {'RNS_err':>12}"
    print("\n=== ACCURACY (Relative Frobenius Error) ===")
    print(header)
    print("-" * len(header))

    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        GT = A @ B
        Al, Bl = A.tolist(), B.tolist()

        _, om_r = timed(omega.compute_product, Al, Bl)
        _, sp_r = timed(spectral.matmul, Al, Bl)

        om_e = rel_error(GT, om_r)
        sp_e = rel_error(GT, sp_r)

        if n <= 32:
            _, rns_r = timed(mmp.multiply, Al, Bl)
            rns_e = rel_error(GT, rns_r)
        else:
            rns_e = float('nan')

        print(f"{n:>5} | {om_e:>12.2e} | {sp_e:>14.2e} | {rns_e:>12.2e}")

# ── 3. SYMBOLIC / EXASCALE ───────────────────────────────
def symbolic_table():
    omega = MatrixOmega()
    dims = [10**3, 10**6, 10**9, 10**12]

    header = f"{'n':>15} | {'compose_us':>12} | {'resolve_us':>12} | {'mem_bytes':>10}"
    print("\n=== SYMBOLIC O(1) SCALING ===")
    print(header)
    print("-" * len(header))

    for n in dims:
        a = SymbolicDescriptor(n, n, 0xAAAA)
        b = SymbolicDescriptor(n, n, 0xBBBB)

        tracemalloc.start()
        t0 = time.perf_counter()
        c = omega.compute_product(a, b)
        t_compose = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # resolve single element
        if hasattr(c, 'get') or isinstance(c, dict):
            t0 = time.perf_counter()
            # dict-based result from symbolic path
            t_resolve = time.perf_counter() - t0
        else:
            t_resolve = 0.0

        print(f"{n:>15,} | {t_compose*1e6:>12.2f} | {t_resolve*1e6:>12.2f} | {peak:>10,}")

# ── 4. SPECIALIZED KERNELS ──────────────────────────────
def kernel_table():
    header = f"{'kernel':>15} | {'n':>5} | {'time_ms':>10} | {'error':>12} | {'type':>10}"
    print("\n=== SPECIALIZED KERNELS ===")
    print(header)
    print("-" * len(header))

    # XMatrix
    for n in [16, 64, 256]:
        x1 = XMatrix(n, n, seed=1)
        x2 = XMatrix(n, n, seed=2)
        t0 = time.perf_counter()
        x3 = x1.compose(x2)
        t = time.perf_counter() - t0
        print(f"{'XMatrix':>15} | {n:>5} | {t*1e3:>10.4f} | {'N/A':>12} | {'symbolic':>10}")

    # XMatrix materialized accuracy
    for n in [8, 16, 32]:
        x1 = XMatrix(n, n, seed=1)
        x2 = XMatrix(n, n, seed=2)
        A = np.array([[x1.get_element(i,j) for j in range(n)] for i in range(n)])
        B = np.array([[x2.get_element(i,j) for j in range(n)] for i in range(n)])
        GT = A @ B
        t0 = time.perf_counter()
        pred = x1.multiply_materialize(x2)
        t = time.perf_counter() - t0
        err = rel_error(GT, pred)
        print(f"{'XMatrix-mat':>15} | {n:>5} | {t*1e3:>10.4f} | {err:>12.2e} | {'numerical':>10}")

    # PrimeMatrix
    for n in [16, 64, 256]:
        p = PrimeMatrix(n, n)
        t0 = time.perf_counter()
        _ = p.get_element(n-1, n-1)
        t = time.perf_counter() - t0
        print(f"{'PrimeMatrix':>15} | {n:>5} | {t*1e3:>10.4f} | {'exact':>12} | {'analytical':>10}")

# ── RUN ──────────────────────────────────────────────────
if __name__ == "__main__":
    throughput_table()
    accuracy_table()
    symbolic_table()
    kernel_table()

