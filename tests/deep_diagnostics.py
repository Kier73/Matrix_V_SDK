import time
import numpy as np
import math
import sys
import os
from typing import List, Tuple, Dict, Any

# Add SDK paths
sys.path.append(os.path.abspath(os.curdir))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega
from matrix_v_sdk.vl.substrate.acceleration import V_SeriesEngine, G_SeriesEngine, P_SeriesEngine

class DeepDiagnostics:
    def __init__(self):
        self.omega = MatrixOmega()
        self.v_engine = V_SeriesEngine(projection_dim=64)
        self.g_engine = G_SeriesEngine(tile_size=4)

    def diag_spectral_stability(self):
        print("\n--- [DIAGNOSTIC 1] Spectral Stability Sweep (N/D Bound) ---")
        N_values = [64, 128, 256, 512]
        D = 64
        for N in N_values:
            A = np.random.randn(N, N).astype(np.float32)
            B = np.random.randn(N, N).astype(np.float32)
            truth = A @ B
            
            t0 = time.perf_counter()
            res = self.v_engine.multiply(A.tolist(), B.tolist())
            lat = (time.perf_counter() - t0) * 1000
            
            err = np.linalg.norm(truth - np.array(res)) / np.linalg.norm(truth)
            print(f"N={N:3} | D={D} | Error: {err:8.4f} | Latency: {lat:8.2f}ms")
            if err > 0.5:
                print(f"  [!] CRITICAL ACCURACY COLLAPSE at N={N}")

    def diag_inductive_cache_sweep(self):
        print("\n--- [DIAGNOSTIC 2] Inductive Cache Efficiency (Tile Size Sweep) ---")
        tile_sizes = [2, 4, 8, 16]
        N = 256
        # Repetitive pattern to trigger cache
        tile_base = np.random.randn(16, 16).astype(np.float32)
        A = np.tile(tile_base, (N//16, N//16)).tolist()
        B = np.tile(tile_base, (N//16, N//16)).tolist()
        
        for ts in tile_sizes:
            engine = G_SeriesEngine(tile_size=ts)
            # Cold run
            engine.multiply(A, B)
            # Warm run (Measures cache hit performance)
            t0 = time.perf_counter()
            engine.multiply(A, B)
            lat = (time.perf_counter() - t0) * 1000
            print(f"Tile Size={ts:2} | Warm Latency: {lat:8.2f}ms")

    def diag_sidechannel_noise_floor(self):
        print("\n--- [DIAGNOSTIC 3] Sidechannel Noise Floor (Locking Sensitivity) ---")
        # Simulate P-series matrix + Gaussian noise
        N = 128
        A_p = [[P_SeriesEngine.resolve_p_series(i+1, j+1, 2) for j in range(N)] for i in range(N)]
        
        snr_levels = [100, 10, 1, 0.1] # High to Low
        for snr in snr_levels:
            noise_std = 1.0 / snr
            A_noisy = [[x + np.random.normal(0, noise_std) for x in row] for row in A_p]
            
            # Check strategy selection
            strategy = self.omega.auto_select_strategy(A_noisy, A_noisy)
            print(f"SNR={snr:5} | Selected Strategy: {strategy}")
            if strategy != "p_series":
                print(f"  [!] LOCK LOST at SNR={snr}")

if __name__ == "__main__":
    diag = DeepDiagnostics()
    diag.diag_spectral_stability()
    diag.diag_inductive_cache_sweep()
    diag.diag_sidechannel_noise_floor()


