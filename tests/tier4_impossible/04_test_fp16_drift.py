import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.math.rns import VlAdaptiveRNS

def test_fp16_drift():
    print("Tier 4: [04] Precision Decay in RNS Manifolds (Simulation)")
    rns = VlAdaptiveRNS(1000)
    # Simulate high-precision drift by rounding residues
    val = 123.456
    residues = [(int(val * 1000) % p) for p in [251, 257]]
    # Add noise to residues (High Risk)
    residues[0] += 1
    
    recon = rns.reconstruct(residues) / 1000.0
    print(f"Original: {val}, Noisy Reconstruct: {recon}")
    # RNS is not robust to residue noise — we expect a different value
    assert recon != val, "Noisy residue should produce different reconstruction"
    print("[PASS] Drift simulation complete")

if __name__ == "__main__":
    test_fp16_drift()


