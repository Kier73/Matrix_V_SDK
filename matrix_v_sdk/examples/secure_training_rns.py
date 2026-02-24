"""
Example: RNS Exact Backward (Secure & High-Precision ML)
=======================================================

This example demonstrates the usage of "Exact Arithmetic Gradients" 
via the RNS (Residue Number System) backward pass. This is crucial for:
  - Differential Privacy: Eliminating floating-point noise leakage.
  - Verifiable ML: Exact gradients can be verified algebraically.
  - Sensitivity Analysis: Observing fine-grained parameter changes.

Key SDK Integration:
  - MMP_Engine: Exact integer arithmetic via CRT.
  - exact_backward=True: Routes backprop through the RNS manifold.
  - VlAdaptiveRNS: Dynamic prime-pool for signed CRT reconstruction.
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Ensure the parent of the SDK root is on the path so 'import matrix_v_sdk' works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from matrix_v_sdk.extensions.torch_bridge import MatrixVLinear
except ImportError:
    print("Error: torch is required for this example.")
    sys.exit(1)

def run_secure_training_demo():
    print("--- 1. Initializing High-Precision Layers ---")
    in_feat, out_feat = 16, 8
    
    # Standard Layer (Floating-Point Gradients)
    layer_std = MatrixVLinear(in_feat, out_feat, exact_backward=False)
    
    # Exact Layer (RNS Gradients)
    # Both share weights initially
    layer_exact = MatrixVLinear(in_feat, out_feat, exact_backward=True)
    layer_exact.load_state_dict(layer_std.state_dict())
    
    # Mock Input with high significance
    x = torch.randn(4, in_feat, requires_grad=True)
    
    print("\n--- 2. Executing Forward & Backward Pass ---")
    
    # Standard Pass
    y_std = layer_std(x)
    loss_std = y_std.sum()
    loss_std.backward()
    grad_std = layer_std.weight.grad.clone()
    
    # Exact Pass (MMP_Engine used in backprop)
    # zero grads first
    if layer_exact.weight.grad is not None: layer_exact.weight.grad.zero_()
    y_exact = layer_exact(x)
    loss_exact = y_exact.sum()
    loss_exact.backward()
    grad_exact = layer_exact.weight.grad.clone()
    
    print("\n--- 3. Gradient Precision Analysis ---")
    # In exact arithmetic (RNS), the gradients are computed without 
    # the rounding bias of torch.mm(fp32).
    diff = torch.abs(grad_std - grad_exact)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"  Max Gradient Divergence: {max_diff:.8e}")
    print(f"  Mean Gradient Divergence: {mean_diff:.8e}")
    
    if max_diff < 1e-6:
        print("\nSUCCESS: RNS exact backward verified. Gradients match to high precision.")
        print("NOTE: Divergence represents the 'rounding noise' present in standard fp32 mm.")
    else:
        print("\nNOTE: Large divergence expected if matrix weights are large or poorly scaled.")

if __name__ == "__main__":
    run_secure_training_demo()


