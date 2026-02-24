import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from typing import Callable, Optional
import warnings
import sys
import os

# Add SDK to path
sys.path.append(os.path.abspath("."))

from matrix_v_sdk.vl.substrate.matrix import MatrixOmega, InfiniteMatrix, SymbolicDescriptor
from matrix_v_sdk.vl.substrate.acceleration import V_SeriesEngine, MMP_Engine, RH_SeriesEngine

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Complexity model definitions
# ---------------------------------------------------------------------------
COMPLEXITY_MODELS = {
    "O(1)":        lambda n, a, b: np.full_like(n, a, dtype=float),
    "O(log n)":    lambda n, a, b: a * np.log2(np.maximum(n, 1e-9)) + b,
    "O(n)":        lambda n, a, b: a * n + b,
    "O(n log n)":  lambda n, a, b: a * n * np.log2(np.maximum(n, 1e-9)) + b,
    "O(n^1.5)":    lambda n, a, b: a * np.power(n, 1.5) + b,
    "O(n^2)":      lambda n, a, b: a * np.power(n, 2) + b,
    "O(n^2 log n)":lambda n, a, b: a * np.power(n, 2) * np.log2(np.maximum(n, 1e-9)) + b,
    "O(n^2.37)":   lambda n, a, b: a * np.power(n, 2.3727) + b,
    "O(n^2.5)":    lambda n, a, b: a * np.power(n, 2.5) + b,
    "O(n^3)":      lambda n, a, b: a * np.power(n, 3) + b,
    "O(n^4)":      lambda n, a, b: a * np.power(n, 4) + b,
    "O(2^n)":      lambda n, a, b: a * np.power(2.0, np.minimum(n, 60)) + b,
}

# ---------------------------------------------------------------------------
# Core benchmarking
# ---------------------------------------------------------------------------

def benchmark_algorithm(
    algorithm: Callable,
    sizes: list[int],
    input_generator: Optional[Callable] = None,
    repeats: int = 3,
    warmup: int = 1,
) -> tuple[list[int], list[float]]:
    if input_generator is None:
        input_generator = lambda n: np.random.rand(n, n).astype(np.float64).tolist()

    sizes_used, times = [], []

    for n in sizes:
        try:
            data = input_generator(n)
            # Warm-up
            for _ in range(warmup):
                algorithm(data)

            # Timed runs
            run_times = []
            for _ in range(repeats):
                # Only refresh input if it's a generator (to avoid massive memory usage at once)
                test_input = input_generator(n)
                t0 = time.perf_counter()
                algorithm(test_input)
                run_times.append(time.perf_counter() - t0)

            sizes_used.append(n)
            times.append(float(np.median(run_times)))
            print(f"  n={n:>6}  ->  {times[-1]*1000:>10.4f} ms")
        except Exception as e:
            print(f"  n={n:>6}  ->  SKIPPED ({e})")

    return sizes_used, times

# ---------------------------------------------------------------------------
# Complexity fitting
# ---------------------------------------------------------------------------

def fit_complexity(sizes: list[int], times: list[float]) -> dict:
    n = np.array(sizes, dtype=float)
    t = np.array(times, dtype=float)

    n_norm = n / n.max()
    t_norm = t / t.max()

    results = {}
    for label, model in COMPLEXITY_MODELS.items():
        try:
            popt, _ = curve_fit(model, n_norm, t_norm, p0=[1.0, 0.0],
                                maxfev=10000, bounds=([0, -np.inf], [np.inf, np.inf]))
            predicted = model(n_norm, *popt)
            ss_res = np.sum((t_norm - predicted) ** 2)
            ss_tot = np.sum((t_norm - np.mean(t_norm)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            results[label] = max(r2, 0.0)
        except Exception:
            results[label] = 0.0

    try:
        log_n = np.log2(n[n > 1])
        log_t = np.log2(t[n > 1])
        if len(log_n) >= 2:
            slope, _ = np.polyfit(log_n, log_t, 1)
        else:
            slope = float("nan")
    except Exception:
        slope = float("nan")

    sorted_fits = sorted(results.items(), key=lambda x: x[1], reverse=True)
    best_label, best_r2 = sorted_fits[0]
    second_r2 = sorted_fits[1][1] if len(sorted_fits) > 1 else 0.0
    gap = best_r2 - second_r2
    confidence = min(100.0, (best_r2 * 60) + (gap * 400))

    return {
        "best_fit": best_label,
        "confidence": round(confidence, 1),
        "r_squared": round(best_r2, 4),
        "exponent": round(slope, 3) if not math.isnan(slope) else None,
        "all_fits": [(lbl, round(r2, 4)) for lbl, r2 in sorted_fits],
    }

# ---------------------------------------------------------------------------
# Reporting & plotting
# ---------------------------------------------------------------------------

def plot_results(
    sizes: list[int],
    times: list[float],
    result: dict,
    algorithm_name: str = "Algorithm",
    save_path: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Big O Analysis — {algorithm_name}", fontsize=14, fontweight="bold")

    n = np.array(sizes)
    t = np.array(times) * 1000  # → ms

    ax = axes[0]
    ax.plot(n, t, "o-", color="#3a86ff", linewidth=2, markersize=6, label="Measured")
    ax.set_xlabel("Matrix dimension (n)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Raw Timing")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    valid = (np.array(sizes) > 1) & (np.array(times) > 0)
    if any(valid):
        log_n = np.log2(n[valid])
        log_t = np.log2((t / 1000)[valid])
        ax2.plot(log_n, log_t, "s-", color="#8338ec", linewidth=2, markersize=6)
        if result["exponent"] is not None:
            intercept_ref = log_t.mean() - result["exponent"] * log_n.mean()
            t_ref = result["exponent"] * log_n + intercept_ref
            ax2.plot(log_n, t_ref, "--", color="#ff6b6b", linewidth=1.5,
                     label=f"slope ≈ {result['exponent']}")
            ax2.legend()
    ax2.set_xlabel("log₂(n)")
    ax2.set_ylabel("log₂(time in s)")
    ax2.set_title("Log-Log Plot")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        # Create artifacts dir if it doesn't exist (though tools handle this usually, 
        # let's be safe with local paths)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def analyze_complexity(
    algorithm: Callable,
    name: str = "My Algorithm",
    sizes: Optional[list[int]] = None,
    input_generator: Optional[Callable] = None,
    repeats: int = 3,
    save_plot: Optional[str] = None,
) -> dict:
    if sizes is None:
        sizes = [8, 16, 32, 64, 128]

    print(f"\nBenchmarking '{name}'...")
    sizes_used, times = benchmark_algorithm(algorithm, sizes, input_generator, repeats)

    if len(sizes_used) < 2:
        print(f"  [ERROR] {name}: Not enough data points.")
        return {}

    result = fit_complexity(sizes_used, times)
    print(f"  Best Fit: {result['best_fit']} (R²={result['r_squared']}, Conf={result['confidence']}%)")
    
    if save_plot:
        plot_results(sizes_used, times, result, name, save_path=save_plot)

    return result

# ---------------------------------------------------------------------------
# Main Suite
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    artifact_dir = r"C:\Users\kross\.gemini\antigravity\brain\a4ac94e1-b8d5-40c1-9c54-0c2b7913a3e5"
    
    # 1. MatrixOmega (Adaptive)
    omega = MatrixOmega()
    analyze_complexity(
        algorithm=lambda data: omega.compute_product(data, data),
        name="MatrixOmega (Adaptive)",
        sizes=[8, 16, 32, 48, 64, 72, 96, 128],
        save_plot=os.path.join(artifact_dir, "complexity_omega.png")
    )

    # 2. InfiniteMatrix (Symbolic)
    def infinite_algo(n):
        desc = SymbolicDescriptor(n, n, 0x123)
        inf_mat = InfiniteMatrix(desc)
        # Force a lazy matmul, then resolve one element to check timing behavior
        # Note: True O(1) if just the matmul is timed, but resolve is O(1) too.
        res_mat = inf_mat.matmul(inf_mat)
        return res_mat[(0, 0)]

    analyze_complexity(
        algorithm=lambda n: infinite_algo(n),
        name="InfiniteMatrix (Symbolic)",
        sizes=[100, 1000, 10000, 100000, 1000000],
        input_generator=lambda n: n, # Pass n directly
        save_plot=os.path.join(artifact_dir, "complexity_infinite.png")
    )

    # 3. V_Series (Spectral Projection)
    v_engine = V_SeriesEngine(epsilon=0.2)
    analyze_complexity(
        algorithm=lambda data: v_engine.multiply(data, data),
        name="V_Series (Spectral)",
        sizes=[32, 64, 128, 256, 512],
        save_plot=os.path.join(artifact_dir, "complexity_spectral.png")
    )

    # 4. MMP_Engine (RNS)
    mmp = MMP_Engine()
    analyze_complexity(
        algorithm=lambda data: mmp.multiply(data, data),
        name="MMP_Engine (RNS)",
        sizes=[8, 16, 24, 32, 40], # RNS is slow in Python simulation
        save_plot=os.path.join(artifact_dir, "complexity_mmp.png")
    )

    print("\nBenchmarking complete. Plots saved to artifacts directory.")

