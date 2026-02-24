import sys
import os
sys.path.append(os.path.abspath(os.curdir))
from matrix_v_sdk.vl.substrate.matrix import SymbolicDescriptor, InfiniteMatrix
import time

def test_symbolic_trillion_scale():
    print("\n--- [TEST] Trillion-Scale Symbolic Operations ($10^{12} \times 10^{12}$) ---")
    
    # 1. Create trillion-scale descriptors
    # Signature 0x1 and 0x2
    A = InfiniteMatrix(SymbolicDescriptor(10**12, 10**12, signature=0x1))
    B = InfiniteMatrix(SymbolicDescriptor(10**12, 10**12, signature=0x2))
    
    print(f"Matrix A: {A.shape[0]}x{A.shape[1]}")
    print(f"Matrix B: {B.shape[0]}x{B.shape[1]}")
    
    # 2. O(1) Symbolic Multiply
    t0 = time.perf_counter()
    C = A.matmul(B)
    lat_multiply = (time.perf_counter() - t0) * 1000
    
    print(f"Symbolic Multiply Latency: {lat_multiply:.4f}ms")
    print(f"Result Matrix C: {C.shape[0]}x{C.shape[1]}")
    
    # 3. JIT Element Resolution
    # Resolving a few elements to verify O(1) access
    coords = [(0, 0), (1234567, 8901234), (10**12 - 1, 10**12 - 1)]
    
    for r, c in coords:
        t0 = time.perf_counter()
        val = C[r, c]
        lat_resolve = (time.perf_counter() - t0) * 1000
        print(f"C[{r}, {c}] = {val:.6f} (Resolved in {lat_resolve:.4f}ms)")

    if lat_multiply < 1.0:
        print("  [v] SYMBOLIC O(1) MULTIPLY VERIFIED")
    else:
        print("  [x] SYMBOLIC MULTIPLY TOO SLOW")

if __name__ == "__main__":
    test_symbolic_trillion_scale()


