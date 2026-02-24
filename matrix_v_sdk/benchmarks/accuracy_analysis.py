import time
import math
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import os

# Import from the integrated SDK using absolute package paths
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega, XMatrix, PrimeMatrix
from matrix_v_sdk.vl.substrate.v_matrix import VMatrix
from matrix_v_sdk.vl.substrate.acceleration import MMP_Engine

warnings.filterwarnings("ignore")

def calculate_relative_error(gt, pred):
    gt = np.array(gt)
    pred = np.array(pred)
    norm_diff = np.linalg.norm(gt - pred)
    norm_gt = np.linalg.norm(gt)
    return norm_diff / (norm_gt + 1e-9)

def benchmark_accuracy(engine_func, sizes, repeats=3):
    results = []
    print(f"Testing {engine_func.__name__}...")
    for n in sizes:
        errors = []
        for _ in range(repeats):
            A = np.random.rand(n, n)
            B = np.random.rand(n, n)
            GT = A @ B
            try:
                A_list = A.tolist()
                B_list = B.tolist()
                PRED = engine_func(A_list, B_list)
                err = calculate_relative_error(GT, PRED)
                errors.append(err)
            except Exception as e:
                print(f"  n={n} failed: {e}")
                continue
        if errors:
            avg_err = np.mean(errors)
            results.append((n, avg_err))
            print(f"  n={n:>4}  ->  Avg Rel Error: {avg_err:.6f}")
    return results

if __name__ == "__main__":
    artifact_dir = r"C:\Users\kross\.gemini\antigravity\brain\a4ac94e1-b8d5-40c1-9c54-0c2b7913a3e5"
    sizes = [16, 32, 64, 128, 256]
    acc_results = {}

    v_spec = VMatrix(mode="spectral")
    acc_results["Spectral"] = benchmark_accuracy(lambda A, B: v_spec.matmul(A, B), sizes)

    mmp = MMP_Engine()
    acc_results["RNS"] = benchmark_accuracy(lambda A, B: mmp.multiply(A, B), [8, 16, 32])

    print("Testing XMatrix Materialization Accuracy...")
    x_results = []
    for n in [8, 16, 32, 64]:
        x1 = XMatrix(n, n, seed=1)
        x2 = XMatrix(n, n, seed=2)
        A_np = np.array([[x1.get_element(i, j) for j in range(n)] for i in range(n)])
        B_np = np.array([[x2.get_element(i, j) for j in range(n)] for i in range(n)])
        GT = A_np @ B_np
        PRED = x1.multiply_materialize(x2)
        err = calculate_relative_error(GT, PRED)
        x_results.append((n, err))
        print(f"  n={n:>4}  ->  Avg Rel Error: {err:.12f}")
    acc_results["XMatrix"] = x_results

    # Visualization
    plt.figure(figsize=(10, 6))
    for name, data in acc_results.items():
        if not data: continue
        s, e = zip(*data)
        plt.plot(s, e, 'o-', label=name)
    plt.title("Accuracy Complexity: Relative Error vs. Matrix Size (n)")
    plt.xlabel("n")
    plt.ylabel("Rel Error")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.savefig(os.path.join(artifact_dir, "accuracy_complexity.png"))
    plt.close()
    print("\nAccuracy Analysis Complete.")

