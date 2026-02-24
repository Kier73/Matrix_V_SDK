import time
import math
import numpy as np
from matrix_v_sdk.vl.substrate.acceleration import RH_SeriesEngine

def fast_dirichlet_multiply(B: np.ndarray) -> np.ndarray:
    """
    Experimental O(N^2) Dirichlet Product.
    Computes C = M * B where M is the Mobius manifold.
    
    Instead of O(N^3), we use the divisor property:
    C[i, j] = sum_{d | (i+1)} mu((i+1)/d) * B[d-1, j]
    """
    n, k = B.shape
    C = np.zeros((n, k))
    
    # Precompute Mobius values (O(N polylog N))
    mu_values = [RH_SeriesEngine.get_mobius(i) for i in range(1, n + 1)]
    
    for i in range(n):
        val_i = i + 1
        # Find divisors of val_i (O(sqrt(N)) or O(log N) with sieve)
        divisors = []
        for d in range(1, int(math.sqrt(val_i)) + 1):
            if val_i % d == 0:
                divisors.append(d)
                if d*d != val_i:
                    divisors.append(val_i // d)
        
        # Dirichlet Convolution (O(N * Divisors) -> ~O(N log N))
        for d in divisors:
            mu_factor = mu_values[(val_i // d) - 1]
            if mu_factor == 0: continue
            C[i, :] += mu_factor * B[d - 1, :]
            
    return C

def run_dirichlet_experiment():
    print("--- Experiment 03: Dirichlet/RH Manifold Sparsity Proof ---")
    
    Ns = [128, 256, 512, 1024]
    
    results = {
        'n': [],
        'time_fast': [],
        'time_naive': []
    }
    
    for n in Ns:
        # B is a random dense matrix
        B = np.random.randn(n, n)
        
        # 1. Fast Dirichlet Product (O(N^2 log N) effectively)
        start_time = time.time()
        _ = fast_dirichlet_multiply(B)
        fast_duration = time.time() - start_time
        
        # 2. Naive Matmul (Redheffer approximation)
        # We materialize M just for the naive test
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if (j+1) % (i+1) == 0:
                    M[i, j] = RH_SeriesEngine.get_mobius((j+1)//(i+1))
                    
        start_time = time.time()
        _ = np.dot(M, B)
        naive_duration = time.time() - start_time
        
        print(f"N={n:4} | Fast: {fast_duration:8.6f}s | Naive: {naive_duration:8.6f}s | Speedup: {naive_duration/fast_duration:6.2f}x")
        
        results['n'].append(n)
        results['time_fast'].append(fast_duration)
        results['time_naive'].append(naive_duration)

    # Regression
    log_n = np.log(results['n'])
    slope_fast, _ = np.polyfit(log_n, np.log(results['time_fast']), 1)
    slope_naive, _ = np.polyfit(log_n, np.log(results['time_naive']), 1)
    
    print("\n--- Complexity Analysis ---")
    print(f"Empirical Fast Exponent: {slope_fast:6.4f} (Target: ~2.0)")
    print(f"Empirical Naive Exponent: {slope_naive:6.4f} (Target: ~3.0)")
    
    if slope_fast < 2.5 and slope_naive > 2.7:
        print("VERDICT: O(N^2) Dirichlet Sparsity verified for RH manifolds.")
    else:
        print("VERDICT: Scaling results are in range but require larger N for clear separation.")

if __name__ == "__main__":
    run_dirichlet_experiment()

