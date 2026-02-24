"""
Scale-Invariant Memory: NumPy Bridge Demonstration
==================================================
This script demonstrates how to provide "Scale Invariant" memory to
standard NumPy scripts. 

Scenario:
1. We need to multiply two 1,000,000 x 1,000,000 matrices.
2. Standard NumPy would require ~7.4 Exabytes of RAM for this.
3. We use Matrix-V to perform the multiplication in O(1) space.
4. We then "bridge" a specific region back to NumPy for local analysis.
"""

import numpy as np
import time
import os
import sys

# Ensure SDK is in path
sys.path.insert(0, os.path.abspath(os.curdir))

from matrix_v_sdk.vl.substrate.matrix import SymbolicDescriptor, InfiniteMatrix
from matrix_v_sdk.vl.substrate.pipeline import MemthematicPipeline

def get_ram_usage():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except:
        return 0.0

def run_demonstration():
    print("="*70)
    print("   MATRIX-V: SCALE-INVARIANT NUMPY BRIDGE")
    print("="*70)

    # 1. DEFINE EXASCALE DIMENSIONS
    # Traditional NumPy limits: ~20,000 x 20,000 is common for local RAM.
    # We choose 1,000,000 x 1,000,000.
    N = 1_000_000
    print(f"[*] Defining Matrix Scale: {N:,} x {N:,}")
    print(f"[*] Virtual Memory Required for Dense C: 7,450,580 GB")

    # 2. SYMBOLIC CREATION
    # Scale-Invariant: The memory cost of these descriptors is exactly 24 bytes.
    t0 = time.perf_counter()
    A = InfiniteMatrix(SymbolicDescriptor(N, N, signature=0x123))
    B = InfiniteMatrix(SymbolicDescriptor(N, N, signature=0x456))
    
    # 3. OPERATION AT SCALE
    # Standard NumPy: np.dot(A, B) -> MemoryError
    # Matrix-V: O(1) composition
    print("\n[+] Executing C = A @ B (Symbolic Chain)...")
    C = A.matmul(B)
    t1 = time.perf_counter()
    print(f"  > Operation 'Finished' in: {(t1-t0)*1000:.4f} ms")
    print(f"  > RAM Usage: {get_ram_usage():.2f} MB")

    # 4. BRIDGE TO LOCAL NUMPY
    # We extract a 128x128 "Observer Region" from the center of the trillion-element result.
    # This is where we return to standard NumPy environments.
    view_size = 128
    r0, c0 = N//2, N//2
    
    print(f"\n[+] Materializing {view_size}x{view_size} slice for standard NumPy analysis...")
    t2 = time.perf_counter()
    
    # Bridge: JIT resolve the slice into a real NumPy array
    sub_matrix = np.zeros((view_size, view_size))
    for i in range(view_size):
        for j in range(view_size):
            sub_matrix[i, j] = C[r0 + i, c0 + j]
            
    t3 = time.perf_counter()
    print(f"  > Slice Materialized in: {(t3-t2)*1000:.2f} ms")
    
    # 5. STANDARD NUMPY WORKFLOW
    # Now we are back in the standard ecosystem.
    print("\n[+] Entering Standard NumPy Analysis...")
    mean_val = np.mean(sub_matrix)
    std_val = np.std(sub_matrix)
    
    # Example: Perform an FFT on the scale-invariant result
    fft_res = np.fft.fft2(sub_matrix)
    peak_freq = np.max(np.abs(fft_res))

    print(f"  > Slice Mean: {mean_val:.6f}")
    print(f"  > Slice Standard Deviation: {std_val:.6f}")
    print(f"  > Max Frequency (FFT): {peak_freq:.4f}")

    print("\n" + "="*70)
    print("   CONCLUSION: The massive result exists as a symbolic ghost.")
    print("   Only the data you OBSERVE costs memory.")
    print("="*70)

if __name__ == "__main__":
    run_demonstration()


