import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.extensions.numba_bridge import NumbaBridge

def test_numba():
    print("Tier 2: [05] Numba JIT Kernel Speedup (Smoke)")
    lattice = NumbaBridge.resolve_p_lattice(10, 10, 2)
    assert lattice.shape == (10, 10)
    print("[PASS]")

if __name__ == "__main__":
    test_numba()


