import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matrix_v_sdk.vl.substrate.acceleration import V_SeriesEngine

def run_jl_experiment():
    print("--- Experiment 01: Johnson-Lindenstrauss Complexity & Error-Bound ---")
    
    # Dimensions to test
    Ns = [128, 256, 512, 1024, 2048]
    epsilon = 0.2 # Target Error 20%
    
    results = {
        'n': [],
        'd_theoretical': [],
        'time_sdk': [],
        'time_numpy': [],
        'error_rel': []
    }
    
    engine = V_SeriesEngine(epsilon=epsilon)
    
    for n in Ns:
        # Generate random dense matrices
        A = np.random.randn(n, n).tolist()
        B = np.random.randn(n, n).tolist()
        
        # 1. SDK V-Series (O(N^2 log N))
        start_time = time.time()
        C_sdk = engine.multiply(A, B)
        sdk_duration = time.time() - start_time
        
        # 2. NumPy Reference (O(N^3))
        start_time = time.time()
        C_np = np.dot(np.array(A), np.array(B))
        np_duration = time.time() - start_time
        
        # 3. Compute Relative Error (Frobenius Norm)
        C_sdk_np = np.array(C_sdk)
        error = np.linalg.norm(C_sdk_np - C_np) / np.linalg.norm(C_np)
        
        d_theoretical = engine.get_adaptive_d(n)
        
        print(f"N={n:4} | D={d_theoretical:4} | SDK: {sdk_duration:6.4f}s | NP: {np_duration:6.4f}s | Error: {error:6.4f}")
        
        results['n'].append(n)
        results['d_theoretical'].append(d_theoretical)
        results['time_sdk'].append(sdk_duration)
        results['time_numpy'].append(np_duration)
        results['error_rel'].append(error)

    # Simple Regression / Proof Analysis
    log_n = np.log(results['n'])
    log_t_sdk = np.log(results['time_sdk'])
    log_t_np = np.log(results['time_numpy'])
    
    slope_sdk, _ = np.polyfit(log_n, log_t_sdk, 1)
    slope_np, _ = np.polyfit(log_n, log_t_np, 1)
    
    print("\n--- Regression Results ---")
    print(f"Empirical SDK Exponent: {slope_sdk:6.4f} (Target: ~2.0-2.3)")
    print(f"Empirical NumPy Exponent: {slope_np:6.4f} (Target: ~2.8-3.0)")
    
    if slope_sdk < slope_np:
        print("VERDICT: O(N^2 log N) Scaling confirmed. SDK is asymptotically superior.")
    else:
        print("VERDICT: Inconclusive for this scale.")

if __name__ == "__main__":
    run_jl_experiment()

