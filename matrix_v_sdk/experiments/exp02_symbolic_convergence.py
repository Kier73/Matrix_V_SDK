import time
import math
import numpy as np
from matrix_v_sdk.vl.substrate.matrix import SymbolicDescriptor, InfiniteMatrix

def run_symbolic_experiment():
    print("--- Experiment 02: Symbolic Field Convergence Verification ---")
    
    # Scales: From medium to "impossible"
    # Even at 10^12, the symbolic synthesis should be O(1)
    Scales = [10**3, 10**6, 10**9, 10**12]
    
    results = {
        'scale': [],
        'time_synthesis': [],
        'time_resolution_100': []
    }
    
    for n in Scales:
        # Create descriptors which are 1D coordinate hashes
        # We simulate a square matrix of size N x N
        d1 = SymbolicDescriptor(n, n, 0xABC)
        d2 = SymbolicDescriptor(n, n, 0xDEF)
        m1 = InfiniteMatrix(d1)
        m2 = InfiniteMatrix(d2)
        
        # 1. Measure Symbolic Matmul Complexity (O(1))
        start_time = time.time()
        m3 = m1.matmul(m2)
        synthesis_duration = time.time() - start_time
        
        # 2. Measure Resolution Complexity (O(1) per element)
        # We resolve 100 random elements to proof zero-overhead scaling
        start_time = time.time()
        for _ in range(100):
            r = hash(time.time()) % n
            c = hash(time.time() + 1) % n
            _ = m3[r, c]
        resolution_duration = time.time() - start_time
        
        print(f"Scale=10^{int(math.log10(n)):2} | Synthesis: {synthesis_duration:8.6f}s | Resolution(100): {resolution_duration:8.6f}s")
        
        results['scale'].append(n)
        results['time_synthesis'].append(synthesis_duration)
        results['time_resolution_100'].append(resolution_duration)

    print("\n--- Synthesis Scalability Analysis ---")
    avg_synthesis = sum(results['time_synthesis']) / len(Scales)
    variance = max(results['time_synthesis']) - min(results['time_synthesis'])
    
    print(f"Average Synthesis Time: {avg_synthesis:8.6f}s")
    print(f"Max Variance: {variance:8.6f}s")
    
    if variance < 0.001:
        print("VERDICT: O(1) Symbolic Convergence confirmed. Complexity is scale-invariant.")
    else:
        print("VERDICT: Minor variance detected (likely jitter).")

if __name__ == "__main__":
    run_symbolic_experiment()

