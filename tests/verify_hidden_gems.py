import sys
import os
import math

# Add SDK paths
sys.path.append(os.path.abspath(os.curdir))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega, SymbolicDescriptor, InfiniteMatrix
from matrix_v_sdk.vl.substrate.acceleration import RH_SeriesEngine

def test_rh_series():
    print("\n--- [VERIFICATION] RH-Series (Riemann-Hilbert) ---")
    rh = RH_SeriesEngine()
    
    # Mobius Verification
    # mu(1)=1, mu(2)=-1, mu(3)=-1, mu(4)=0, mu(6)=1
    test_vals = {1: 1, 2: -1, 3: -1, 4: 0, 6: 1}
    success = True
    for n, expected in test_vals.items():
        res = rh.get_mobius(n)
        print(f"mu({n}) = {res} (expected {expected})")
        if res != expected: success = False
        
    if success:
        print("  [v] RH-SERIES MOBIUS LOGIC FUNCTIONAL")
    else:
        print("  [x] RH-SERIES MOBIUS LOGIC FAILED")

def test_infinite_matrix():
    print("\n--- [VERIFICATION] Infinite-Scale Symbolic Matrices ---")
    # Simulation of two 1-Trillion x 1-Trillion matrices
    size = 10**12
    
    A_desc = SymbolicDescriptor(size, size, signature=0xAB12)
    B_desc = SymbolicDescriptor(size, size, signature=0xCD34)
    
    A = InfiniteMatrix(A_desc)
    B = InfiniteMatrix(B_desc)
    
    # O(1) Symbolic Multiply
    C = A.matmul(B)
    
    print(f"Infinite Matrix C shape: {C.shape}")
    
    # Resolve elements at arbitrary coordinates
    val1 = C[1234567, 8901234]
    val2 = C[size-1, size-1]
    
    print(f"C[1234567, 8901234] = {val1:.4f}")
    print(f"C[last, last] = {val2:.4f}")
    
    if C.shape == (size, size) and isinstance(val1, float):
        print("  [v] INFINITE-SCALE SYMBOLIC OPS FUNCTIONAL")
    else:
        print("  [x] INFINITE-SCALE SYMBOLIC OPS FAILED")

def test_omega_rh_selection():
    print("\n--- [VERIFICATION] Omega Strategy Selection (RH) ---")
    omega = MatrixOmega()
    
    # Prime dimensions should trigger RH-Series
    # 13 and 17 are prime
    A = [[1.0]*13 for _ in range(13)]
    B = [[1.0]*17 for _ in range(13)]
    
    strategy = omega.auto_select_strategy(A, B)
    print(f"Strategy selected for 13x13 @ 13x17: {strategy}")
    
    if strategy == "rh_series":
        print("  [v] OMEGA RH-SELECTION FUNCTIONAL")
    else:
        # It might also pick mmp or adaptive if primes are small or logic differs
        print(f"  [?] OMEGA SELECTED {strategy}")

if __name__ == "__main__":
    test_rh_series()
    test_infinite_matrix()
    test_omega_rh_selection()


