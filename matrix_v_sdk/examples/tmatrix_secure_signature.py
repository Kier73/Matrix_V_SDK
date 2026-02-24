"""
T-Matrix Industrial Use Case: Secure Holographic Signatures
===========================================================
Demonstrates using RNS residues to generate a bit-exact holographic 
signature of a matrix manifold. This signature is used to verify model 
integrity in zero-trust or decentralized computing environments without 
exposing the full weight DNA.

Theory: Signature = RNS(Gielis(DNA))
"""
import sys
import os
import torch
import time

SDK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SDK_ROOT)

from matrix_v_sdk.vl.substrate.tmatrix import TMatrix

def run_secure_scenario():
    print(">>> SCENARIO: SECURE HOLOGRAPHIC SIGNATURE")
    print("Task: Verify weight manifold integrity between two nodes.")
    
    # Node A: Canonical DNA
    shape = (1024, 1024)
    dna_a = [4.0, 1.0, 1.0, 0.5, 0.5, 0.5]
    tm_a = TMatrix(shape, params=dna_a)
    
    # Node B: Received DNA (Correct)
    tm_b_correct = TMatrix(shape, params=dna_a)
    
    # Node C: Compromised DNA (Adversarial shift)
    dna_c = list(dna_a)
    dna_c[0] += 1e-7 # Tiny adversarial perturbation
    tm_c_malicious = TMatrix(shape, params=dna_c)
    
    print("  Generating Holographic Signatures via RNS Basis...")
    
    start = time.time()
    sig_a = tm_a.get_rns_signature(count=32) # Using full 32-prime basis for security
    sig_b = tm_b_correct.get_rns_signature(count=32)
    sig_c = tm_c_malicious.get_rns_signature(count=32)
    end = time.time()
    
    print(f"  [NODE A] Canonical: {hex(sig_a)}")
    print(f"  [NODE B] Received:  {hex(sig_b)}")
    print(f"  [NODE C] Malicious: {hex(sig_c)}")
    
    valid_b = "MATCH" if sig_a == sig_b else "MISMATCH"
    valid_c = "MATCH" if sig_a == sig_c else "MISMATCH"
    
    print(f"\n  Integrity Check Node B: {valid_b}")
    print(f"  Integrity Check Node C: {valid_c}")
    
    print(f"  Signature Generation Time: {(end-start)*1000:.2f}ms")
    
    if sig_a == sig_b and sig_a != sig_c:
        print("\n  [SUCCESS] Holographic RNS verification is sensitive to 1e-7 DNA shifts.")
    else:
        print("\n  [FAIL] Integrity sensitivity insufficient.")

if __name__ == "__main__":
    run_secure_scenario()


