"""
Example: Adaptive LLM Projection (PyTorch + CUDA Bridge)
======================================================

This example demonstrates using the Matrix SDK to accelerate the 
projection layers of a Transformer model. It uses the MatrixVLinear 
layer, which automatically routes compute between CPU (symbolic/exact) 
and GPU (bulk cuBLAS) based on the learned strategy.

Key SDK Integration:
  - cuda_bridge.py: Card-agnostic NVIDIA acceleration
  - torch_bridge.py: MatrixVLinear drop-in replacement
  - StrategyCache: Learns optimal device-routing for specific shapes
"""
import sys
import os
import torch
import torch.nn as nn
import time

# Ensure the parent of the SDK root is on the path so 'import matrix_v_sdk' works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from matrix_v_sdk.extensions.torch_bridge import MatrixVLinear
    from matrix_v_sdk.extensions.cuda_bridge import CUDADeviceInfo
except ImportError:
    print("Error: torch is required for this example. Run 'pip install torch'")
    sys.exit(1)

def benchmark_transformer_layer():
    print("--- GPU Hardware Audit ---")
    info = CUDADeviceInfo()
    print(info)
    
    # Define a standard Transformer projection size 
    # (e.g., Llama-2 7B: d_model=4096)
    in_feat = 1024
    out_feat = 1024
    batch_size = 16
    seq_len = 128
    
    print(f"\nConstructing MatrixVLinear Layer ({in_feat} -> {out_feat})...")
    # This layer uses the Matrix SDK backend
    vl_linear = MatrixVLinear(in_feat, out_feat)
    
    # Standard PyTorch layer for comparison
    torch_linear = nn.Linear(in_feat, out_feat)
    torch_linear.load_state_dict(vl_linear.state_dict()) # Match weights
    
    # Mock Input (Batch, SeqLen, InFeat)
    x = torch.randn(batch_size, seq_len, in_feat)
    
    if info.available:
        print("\nSwitching to CUDA device...")
        vl_linear = vl_linear.to("cuda")
        torch_linear = torch_linear.to("cuda")
        x = x.to("cuda")

    # 1. First Forward Pass (Warming up the adaptive strategy cache)
    print("\nPass 1: Warming up Strategy Cache (Adaptive Classification)...")
    start = time.perf_counter()
    y_vl = vl_linear(x)
    end = time.perf_counter()
    print(f"  Matrix-V Latency: {(end - start)*1000:.2f}ms")
    
    # 2. Repeated Passes (Executing learned optimal strategy)
    print("\nPass 2-10: Executing Promoted Strategy...")
    timings = []
    for _ in range(9):
        start = time.perf_counter()
        _ = vl_linear(x)
        if info.available: torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    
    avg_vl = sum(timings) / len(timings)
    print(f"  Matrix-V Avg Latency: {avg_vl:.2f}ms")
    
    # 3. Standard PyTorch Baseline
    timings_pt = []
    for _ in range(10):
        start = time.perf_counter()
        _ = torch_linear(x)
        if info.available: torch.cuda.synchronize()
        timings_pt.append((time.perf_counter() - start) * 1000)
    
    avg_pt = sum(timings_pt) / len(timings_pt)
    print(f"  Standard Torch Avg Latency: {avg_pt:.2f}ms")
    
    speedup = avg_pt / avg_vl if avg_vl > 0 else 0
    print(f"\nSummary: Matrix-V provides a {speedup:.2x} speedup via adaptive routing.")

if __name__ == "__main__":
    benchmark_transformer_layer()


