import os
import sys
import numpy as np

# Add SDK parent to path
SDK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDK_PARENT = os.path.dirname(SDK_ROOT)
sys.path.insert(0, SDK_PARENT)

from matrix_v_sdk.vl.substrate.matrix import SymbolicDescriptor, MatrixOmega

def audit_parity():
    print("--- PARITY AUDIT: SYMBOLIC VS DENSE ---")
    N = 4 # Small enough for dense
    
    # 1. DENSE MULTIPLY
    A_dense = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] # Identity
    B_dense = [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]
    
    omega = MatrixOmega()
    C_dense = omega.naive_multiply(A_dense, B_dense)
    print(f"Dense Result[0,0]: {C_dense[0][0]}")
    
    # 2. SYMBOLIC MULTIPLY
    A_sym = SymbolicDescriptor(N, N, signature=0x1)
    B_sym = SymbolicDescriptor(N, N, signature=0x2)
    C_sym = A_sym.multiply(B_sym)
    
    val_sym = C_sym.resolve(0, 0)
    print(f"Symbolic Result[0,0]: {val_sym:.4f}")
    
    print("\n[CONCLUSION]")
    if abs(C_dense[0][0] - val_sym) > 1e-5:
        print("!!! SCRUTINY ALERT !!!")
        print("Symbolic result is NOT numerically equal to Dense result.")
        print("The Symbolic layer is an 'Algebraic Isomorphism', not a 'Numerical Identity'.")
        print("Recommendation: Do not claim 'Numerical Parity' for the O(1) trillion-scale mode.")
    else:
        print("Symbolic and Dense are numerically aligned.")

if __name__ == "__main__":
    audit_parity()


