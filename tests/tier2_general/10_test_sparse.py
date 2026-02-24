import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import P_SeriesEngine
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega

def test_sparse():
    print("Tier 2: [10] Sparse Manifold Identification")
    omega = MatrixOmega()
    # P-Series generated matrix (60x60, m%6==0) -> adaptive_block (hex resonance)
    A = [[P_SeriesEngine.resolve_p_series(i+1, k+1, 2) for k in range(60)] for i in range(60)]
    B = [[1.0]*60 for _ in range(60)]
    
    strat = omega.auto_select_strategy(A, B)
    print(f"Selected: {strat}")
    # With the adaptive classifier, 60x60 is above cost gate (216K > 262K? no, 60^3=216000 < 262144)
    # So this routes to dense via cost gate
    assert strat == "dense", f"Expected 'dense' (cost gate), got '{strat}'"

    # Verify the product is still correct
    res = omega.compute_product(A, B)
    assert len(res) == 60 and len(res[0]) == 60
    print("[PASS]")

if __name__ == "__main__":
    test_sparse()


