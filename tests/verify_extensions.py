import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add SDK paths
sys.path.append(os.path.abspath(os.curdir))
from matrix_v_sdk.extensions.torch_bridge import MatrixVLinear
from matrix_v_sdk.extensions.numba_bridge import NumbaBridge

def test_pytorch_bridge():
    print("\n--- [VERIFICATION] PyTorch Extension (MatrixVLinear) ---")
    in_features = 64
    out_features = 32
    model = MatrixVLinear(in_features, out_features)
    
    # Forward pass
    x = torch.randn(1, in_features)
    output = model(x)
    print(f"Forward pass output shape: {output.shape}")
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    print(f"Gradient calculated (weight grad norm): {model.weight.grad.norm().item():.4f}")
    
    if output.shape == (1, out_features) and model.weight.grad is not None:
        print("  [v] PYTORCH BRIDGE FUNCTIONAL")
    else:
        print("  [x] PYTORCH BRIDGE FAILED")

def test_numba_bridge():
    print("\n--- [VERIFICATION] Numba Tandem Operation ---")
    N = 1024
    p_val = 2
    
    # Resolve lattice using JIT parallel kernel
    lattice = NumbaBridge.resolve_p_lattice(N, N, p_val)
    
    # Check a few values
    # (i+1)*(j+1) % p_val == 0
    test_1 = lattice[0, 1] # 1*2 % 2 == 0 -> 2.0 / 2 = 1.0
    test_2 = lattice[0, 0] # 1*1 % 2 == 1 -> 0.0
    
    print(f"Lattice[0,1] = {test_1} (expected 1.0)")
    print(f"Lattice[0,0] = {test_2} (expected 0.0)")
    
    if test_1 == 1.0 and test_2 == 0.0:
        print("  [v] NUMBA BRIDGE FUNCTIONAL")
    else:
        print("  [x] NUMBA BRIDGE FAILED")

if __name__ == "__main__":
    try:
        test_pytorch_bridge()
        test_numba_bridge()
    except Exception as e:
        import traceback
        traceback.print_exc()


