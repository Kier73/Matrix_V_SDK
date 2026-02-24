import sys
import os
import numpy as np

# Add SDK root
sdk_root = os.path.abspath(os.getcwd())
sys.path.append(sdk_root)

try:
    from matrix_v_sdk.vl.substrate.matrix import MatrixOmega, SymbolicDescriptor
    from matrix_v_sdk.vl.substrate.vld_holographic import Hypervector
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_poc_integration():
    omega = MatrixOmega(seed=0x123)
    
    print("--- Test 1: Trinity Consensus (Holographic Path) ---")
    trinity_req = {"trinity": True, "law": "Gravity", "intent": "Falling"}
    event_data = [1.0, 0.0, 9.8] # Ground event
    res = omega.compute_product(trinity_req, event_data)
    print(f"  Result Hypervector: {res.label}")
    print(f"  Result Signature: {hex(res.signature())}")

    print("\n--- Test 2: Exascale G-Dynamics (Symbolic Path) ---")
    # Mocking objects with 'signature' and 'rows'/'cols'
    class MockGDescriptor:
        def __init__(self, r, c, sig):
            self.rows, self.cols, self.signature = r, c, sig
    
    desc_a = MockGDescriptor(10**12, 10**12, 0xAAAAA)
    desc_b = MockGDescriptor(10**12, 10**12, 0xBBBBB)
    
    res_desc = omega.compute_product(desc_a, desc_b)
    print(f"  Result Meta-Descriptor: {res_desc}")
    print(f"  Unified Signature: {hex(res_desc['signature'])}")

if __name__ == "__main__":
    test_poc_integration()

