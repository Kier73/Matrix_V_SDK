import time
import numpy as np
import math
import sys
import os

# Add SDK paths
sys.path.append(os.path.abspath(os.curdir))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega
from matrix_v_sdk.vl.substrate.acceleration import V_SeriesEngine, G_SeriesEngine, SidechannelDetector

def test_adaptive_spectral():
    print("\n--- [VERIFICATION 1] Adaptive Spectral Stability (N=512) ---")
    v_engine = V_SeriesEngine(epsilon=0.05) # Tighter epsilon
    N = 512
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    truth = A @ B
    
    t0 = time.perf_counter()
    res = v_engine.multiply(A.tolist(), B.tolist())
    lat = (time.perf_counter() - t0) * 1000
    
    err = np.linalg.norm(truth - np.array(res)) / np.linalg.norm(truth)
    adaptive_d = v_engine.get_adaptive_d(N)
    print(f"N={N} | Adaptive D={adaptive_d} | Error: {err:.4f} | Latency: {lat:.2f}ms")
    if err < 0.2: # Significant improvement over 1.0+ error
        print("  [v] STABILITY ACHIEVED")
    else:
        print("  [x] STABILITY FAILED")

def test_fuzzy_locking():
    print("\n--- [VERIFICATION 2] Fuzzy Sidechannel Locking ---")
    detector = SidechannelDetector()
    
    # Original target
    target = [0.1, 0.5, -0.2, 0.8, -0.5, 0.3, 0.7, -0.1]
    # Noisy signal (96% similarity)
    noisy_signal = [x + np.random.normal(0, 0.05) for x in target]
    
    seed = detector.probe_stream_block(noisy_signal)
    print(f"Signal Similarity: {detector.confidence:.4f}")
    if seed:
        print(f"  [v] LOCK ACHIEVED: Seed {hex(seed)}")
    else:
        print("  [x] LOCK FAILED")

def test_rust_cache_performance():
    print("\n--- [VERIFICATION 3] Rust-Backed Inductive Caching ---")
    g_engine = G_SeriesEngine(tile_size=16)
    N = 256
    tile_base = np.random.randn(16, 16).astype(np.float32)
    A = np.tile(tile_base, (N//16, N//16)).tolist()
    B = np.tile(tile_base, (N//16, N//16)).tolist()
    
    # Cold run
    g_engine.multiply(A, B)
    
    # Warm run (Cache Hit)
    t0 = time.perf_counter()
    g_engine.multiply(A, B)
    lat = (time.perf_counter() - t0) * 1000
    print(f"Warm Latency (Rust Cache): {lat:.2f}ms")
    if lat < 100:
        print("  [v] PERFORMANCE GOAL MET")
    else:
        print("  [x] PERFORMANCE GOAL FAILED")

if __name__ == "__main__":
    try:
        test_adaptive_spectral()
        test_fuzzy_locking()
        test_rust_cache_performance()
    except Exception as e:
        import traceback
        traceback.print_exc()


