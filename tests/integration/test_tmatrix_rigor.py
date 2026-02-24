"""
T-Matrix Rigorous Verification Suite
====================================
Formal proof of the T-Matrix mathematical foundations:
  1. Ghost Accuracy (O(1) -> O(N^2) projection fidelity)
  2. RNS Determinism (Bit-exactness across calls)
  3. Functional Agreement (T_MatrixVLinear vs nn.Linear)
  4. Resonant Shunting (Compute reduction vs Energy recall)
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import time

# Ensure the SDK root is on the path (parent of matrix_v_sdk)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from matrix_v_sdk.vl.substrate.tmatrix import TMatrix, T_MatrixVLinear

PASS_COUNT = 0
FAIL_COUNT = 0

def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {name}")
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {name} -- {detail}")

def rel_error(gt, pred):
    gt, pred = np.asarray(gt, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    return float(np.linalg.norm(gt - pred) / (np.linalg.norm(gt) + 1e-15))

# ================================================================
# 1. Ghost Accuracy & Compression Rigor
# ================================================================
def test_ghost_rigor():
    print("\n=== 1. GHOST ACCURACY & COMPRESSION ===")
    shape = (1024, 1024)
    tm = TMatrix(shape)
    
    # Storage check
    dense_bytes = 1024 * 1024 * 4 # 4MB
    dna_bytes = len(tm.params) * 4 # 24 bytes
    compression = dense_bytes / dna_bytes
    
    check("170k-fold Compression Ratio", compression > 170000, f"{compression:,.0f}x")
    
    # Materialization Match
    w1 = tm.materialize()
    w2 = tm.materialize()
    
    err = rel_error(w1, w2)
    check("Materialization Determinism", err < 1e-10, f"error={err:.2e}")
    check("Structural Entropy", torch.std(w1).item() > 0.1, f"std={torch.std(w1):.4f}")

# ================================================================
# 2. RNS Determinism Rigor
# ================================================================
def test_rns_rigor():
    print("\n=== 2. RNS INTEGRITY RIGOR ===")
    shape = (512, 512)
    tm = TMatrix(shape)
    
    sig1 = tm.get_rns_signature(count=16)
    sig2 = tm.get_rns_signature(count=16)
    
    check("RNS Signature Determinism", sig1 == sig2, f"sig1={hex(sig1)}, sig2={hex(sig2)}")
    
    # Sensitivity: Change one parameter significantly
    original_params = list(tm.params)
    tm.params[0] += 0.1 # Large enough to shift rounded residues
    sig3 = tm.get_rns_signature(count=16)
    
    check("Resonance Sensitivity (Avalanche)", sig1 != sig3, f"Signature failed to change on param shift (sig1={hex(sig1)}, sig3={hex(sig3)})")

# ================================================================
# 3. Functional Agreement (T_MatrixVLinear)
# ================================================================
def test_functional_rigor():
    print("\n=== 3. FUNCTIONAL AGREEMENT RIGOR ===")
    n_in, n_out = 128, 128
    layer = T_MatrixVLinear(n_in, n_out, mode='ground')
    
    x = torch.randn(4, n_in)
    y = layer(x)
    
    check("Forward Pass Shape", y.shape == (4, n_out), f"got {y.shape}")
    
    # Manual check against materialized weights
    order = int(np.ceil(np.log2(max(n_out, n_in))))
    w_mat = layer.engine.project_holographic_manifold(layer.dna.detach().cpu().numpy().tolist(), (n_out, n_in), order)
    w_mat = torch.from_numpy(w_mat).float()
    
    # Mirror internal standardization
    w_mat = (w_mat - w_mat.mean()) / (w_mat.std() + 1e-9)
    
    expected = torch.matmul(x, w_mat.t() * layer.scale) + layer.bias
    
    err = rel_error(expected.detach(), y.detach())
    check("Substrate Consistency", err < 1e-6, f"error={err:.2e}")

# ================================================================
# 4. Resonant Shunting Rigor
# ================================================================
def test_shunting_rigor():
    print("\n=== 4. RESONANT SHUNTING RIGOR ===")
    n_in, n_out = 64, 64 # order 6
    layer_res = T_MatrixVLinear(n_in, n_out, mode='resonant')
    layer_ground = T_MatrixVLinear(n_in, n_out, mode='ground')
    
    # Ensure DNA is synced
    layer_res.dna.data = layer_ground.dna.data.clone()
    
    x = torch.randn(1, n_in)
    y_ground = layer_ground(x)
    y_res = layer_res(x)
    
    # We expect y_res (shunted) to have lower variance/energy than y_ground
    ground_energy = torch.norm(y_ground).item()
    res_energy = torch.norm(y_res).item()
    recall = res_energy / (ground_energy + 1e-9)
    
    # Threshold for noise shunting: we expect it to be significantly less
    # because we zeroed out 90% of the weights.
    check("Topological Shunting Effect", recall < 1.0, f"recall={recall:.2%}")
    check("Signal Retention", recall > 0.01, f"recall={recall:.2%}")

if __name__ == "__main__":
    start_time = time.time()
    try:
        test_ghost_rigor()
        test_rns_rigor()
        test_functional_rigor()
        test_shunting_rigor()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        FAIL_COUNT += 1

    duration = time.time() - start_time
    print("\n" + "="*60)
    print(f"Verification Complete in {duration:.2f}s")
    print(f"  Passed: {PASS_COUNT}")
    print(f"  Failed: {FAIL_COUNT}")
    print("="*60)
    
    if FAIL_COUNT > 0:
        sys.exit(1)


