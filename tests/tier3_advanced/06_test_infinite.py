import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.matrix import SymbolicDescriptor, InfiniteMatrix

def test_infinite():
    print("Tier 3: [06] Trillion-Scale Symbolic Operations")
    size = 10**12
    A = InfiniteMatrix(SymbolicDescriptor(size, size, 0x1))
    B = InfiniteMatrix(SymbolicDescriptor(size, size, 0x2))
    C = A.matmul(B)
    val = C[1234, 5678]
    print(f"C[1234, 5678] = {val}")
    assert isinstance(val, float)
    print("[PASS]")

if __name__ == "__main__":
    test_infinite()


