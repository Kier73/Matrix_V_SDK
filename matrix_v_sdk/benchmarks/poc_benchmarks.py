import time
import math
import hashlib
import struct
import numpy as np
import sys
import os

# Add POC path
poc_path = r"C:\Users\kross\Downloads\Virtual Layer  - POC"
sys.path.append(poc_path)

try:
    from matrix_v_sdk.vld_sdk.matrix import VMatrix, GMatrix, GDescriptor, XMatrix
    from matrix_v_sdk.vld_sdk.holographic import Hypervector, TrinityConsensus
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def benchmark_poc_symbolic():
    print("--- POC Symbolic Benchmarks ---")
    sizes = [100, 1000, 10000, 100000]
    
    # 1. GMatrix O(1) Access
    desc = GDescriptor(10**6, 10**6, 0x12345678)
    g_mat = GMatrix(desc)
    
    print("Testing GMatrix JIT Realization (O(1) Access):")
    for n in sizes:
        start = time.perf_counter()
        # Sample 100 elements
        for _ in range(100):
            r = np.random.randint(0, n)
            c = np.random.randint(0, n)
            _ = g_mat.resolve(r, c)
        end = time.perf_counter()
        print(f"  n={n:<10} | Avg Resolve Time: {(end-start)/100*1000:.6f} ms")

    # 2. XMatrix Semantic Resolve
    print("\nTesting XMatrix Semantic Engine:")
    for n in [100, 1000, 5000]:
        x_mat = XMatrix(n, n, seed=42)
        start = time.perf_counter()
        # Sample 10 elements (HDC is slower in pure Python)
        for _ in range(10):
            r = np.random.randint(0, n)
            c = np.random.randint(0, n)
            _ = x_mat.get_element(r, c)
        end = time.perf_counter()
        print(f"  n={n:<10} | Avg Element Time: {(end-start)/10*1000:.6f} ms")

def benchmark_holographic_consensus():
    print("\n--- Holographic Consensus Benchmarks ---")
    trinity = TrinityConsensus(0x777)
    
    print("Testing Trinity Convergence (Law ^ Intention ^ Event):")
    # This represents O(1) truth resolution
    inputs = [
        ("Gravity", "Falling", 0x111),
        ("Electromagnetism", "Repulsion", 0x222),
        ("QuantumRelativity", "Entanglement", 0x333)
    ]
    
    for law, intent, event in inputs:
        start = time.perf_counter()
        res = trinity.resolve(law, intent, event)
        end = time.perf_counter()
        print(f"  [{law}] -> Result Sig: {hex(res.signature())} | Time: {(end-start)*1000:.4f} ms")

if __name__ == "__main__":
    benchmark_poc_symbolic()
    benchmark_holographic_consensus()

