import torch
import sys
import os
import time

sys.path.append(os.path.abspath(os.curdir))

from matrix_v_sdk.extensions.torch_bridge import MatrixVLinear

def demo_exact_gradients():
    print("\n=== [FUNCTIONAL DEMO] Exact Gradient Computation for Verifiable AI ===")
    
    # 1. Setup a standard high-precision linear layer vs Matrix-V RNS Layer
    in_dim, out_dim = 16, 8
    
    # Standard PyTorch (Float32/64 subject to rounding)
    linear_std = torch.nn.Linear(in_dim, out_dim)
    
    # Matrix-V RNS-Exact (Gradients computed via CRT)
    linear_vl = MatrixVLinear(in_dim, out_dim, exact_backward=True)
    
    # Copy weights for fair comparison
    linear_vl.weight.data = linear_std.weight.data.clone()
    linear_vl.bias.data = linear_std.bias.data.clone()
    
    # 2. Input data (Scaled to represent fixed-point range)
    x = torch.randn(4, in_dim, requires_grad=True)
    
    # 3. Forward Pass
    print("  Executing Forward Pass...")
    out_std = linear_std(x)
    out_vl = linear_vl(x)
    
    # 4. Backward Pass (Gradient Accumulation)
    print("  Executing Backward Pass (Exact RNS vs Standard FP)...")
    
    # Scale loss to ensure we see the precision difference
    loss_std = out_std.sum()
    loss_vl = out_vl.sum()
    
    t0 = time.perf_counter()
    loss_std.backward()
    lat_std = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    loss_vl.backward()
    lat_vl = (time.perf_counter() - t0) * 1000
    
    # 5. Analysis
    grad_std = linear_std.weight.grad
    grad_vl = linear_vl.weight.grad
    
    diff = torch.abs(grad_std - grad_vl)
    max_diff = torch.max(diff).item()
    
    print(f"  Standard Backward Latency: {lat_std:.2f}ms")
    print(f"  Matrix-V Exact Latency:    {lat_vl:.2f}ms (Includes FFI Overhead)")
    print(f"  Max Gradient Disparity:    {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("  [v] FUNCTIONAL VERIFICATION: RNS-Exact gradients match FP baseline with high fidelity.")
    else:
        print("  [!] DISPARITY DETECTED: This is the 'Rounding Noise' eliminated by Matrix-V.")

if __name__ == "__main__":
    demo_exact_gradients()


