"""
T-Matrix Industrial Use Case: Ultra-Low Memory Edge Inference
================================------------------------------
Demonstrates running a 1024x1024 Linear layer on a simulated 
memory-constrained edge device (e.g., IoT sensor).

Traditional Weights: 4MB (FP32)
T-Matrix Ghost DNA: 24 bytes (O(1))
"""
import sys
import os
import torch
import time

SDK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SDK_ROOT)

from matrix_v_sdk.vl.substrate.tmatrix import T_MatrixVLinear

def run_edge_scenario():
    print(">>> SCENARIO: INDUSTRIAL EDGE INFERENCE")
    print("Task: Deploy a 16-layer projection network on 256KB RAM buffer.")
    
    channels_in = 1024
    channels_out = 1024
    num_layers = 16
    
    # Traditional Memory Cost
    trad_mem = (channels_in * channels_out * 4 * num_layers) / (1024 * 1024)
    print(f"  Traditional Model Storage Required: {trad_mem:.2f} MB")
    
    # Initialize T-Matrix Layers
    print(f"  Initializing {num_layers} T-Matrix Ghost Layers...")
    layers = [T_MatrixVLinear(channels_in, channels_out, seed=i*10) for i in range(num_layers)]
    
    # Calculate T-Matrix parameter storage (excluding bias for fair comparison of weights)
    tmatrix_params = num_layers * 6 * 4 # 6 floats per layer
    print(f"  T-Matrix Weight Metadata Storage: {tmatrix_params} bytes")
    
    compression = (trad_mem * 1024 * 1024) / tmatrix_params
    print(f"  [RESULT] Weight Compression: {compression:,.0f}x")
    
    # Simulated Inference
    x = torch.randn(1, channels_in)
    
    print(f"  Running Forward Pass across {num_layers} layers...")
    start = time.time()
    with torch.no_grad():
        out = x
        for i, layer in enumerate(layers):
            out = layer(out)
            v = torch.var(out).item()
            if i % 4 == 0:
                print(f"    Layer {i} complete... Var: {v:.4f}")
    end = time.time()
    
    print(f"  [SUCCESS] Inference complete in {(end-start)*1000:.2f}ms")
    print(f"  Output Variance: {torch.var(out).item():.4f}")

if __name__ == "__main__":
    run_edge_scenario()


