import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.matrix import SymbolicDescriptor

def test_depth():
    print("Tier 4: [03] Deep Descriptor Nesting (Depth=1000)")
    desc = SymbolicDescriptor(10, 10, 0x1)
    for _ in range(1000):
        desc = desc.multiply(SymbolicDescriptor(10, 10, 0x1))
    
    print(f"Final Depth: {desc.depth}")
    assert desc.depth == 1001
    val = desc.resolve(0, 0)
    print(f"Resolved (0,0): {val}")
    print("[PASS]")

if __name__ == "__main__":
    test_depth()


