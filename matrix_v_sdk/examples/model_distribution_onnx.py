"""
Example: Model Distribution & Portability (SafeTensors + ONNX)
=============================================================

This example demonstrates the standard workflow for distributing a 
model trained with the Matrix SDK.

1. Trained weights are saved to .safetensors.
2. The learned StrategyCache (adaptive routing) is embedded as metadata.
3. The model is exported to .onnx for deployment on non-Python runtimes.

Key SDK Integration:
  - safetensors_bridge.py: High-speed IO with cache metadata.
  - onnx_bridge.py: Graph substitution for cross-platform deployment.
"""
import sys
import os
import torch
import torch.nn as nn
import json

# Ensure the parent of the SDK root is on the path so 'import matrix_v_sdk' works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from matrix_v_sdk.extensions.torch_bridge import MatrixVLinear
    from matrix_v_sdk.extensions.safetensors_bridge import save_matrix_v_model, load_matrix_v_model
    from matrix_v_sdk.extensions.onnx_bridge import export_to_onnx
except ImportError:
    print("Error: torch and safetensors are required for this example.")
    sys.exit(1)

def run_distribution_workflow():
    print("--- 1. Constructing & Training Model ---")
    model = nn.Sequential(
        MatrixVLinear(128, 64),
        nn.ReLU(),
        MatrixVLinear(64, 10)
    )
    
    # Simulate some adaptive learning (pumping observations)
    # This will allow the StrategyCache to lock into 'adaptive_block' or 'qmatrix'
    print("Simulating adaptive learning (Warming up local StrategyCache)...")
    dummy_input = torch.randn(1, 128)
    for _ in range(10):
        _ = model(dummy_input)
    
    # --- STEP 2: SAVE TO SAFETENSORS ---
    st_path = "matrix_v_model.safetensors"
    print(f"\n--- 2. Saving to SafeTensors: {st_path} ---")
    # We pass the strategy_cache from one of the layers (or a shared one)
    # to be embedded in the metadata.
    save_matrix_v_model(model, st_path, strategy_cache=model[0].omega.strategy_cache)
    print("  File saved with embedded StrategyCache metadata.")
    
    # --- STEP 3: LOAD INTO NEW INSTANCE ---
    print("\n--- 3. Loading into Fresh Instance ---")
    model_new = nn.Sequential(
        MatrixVLinear(128, 64),
        nn.ReLU(),
        MatrixVLinear(64, 10)
    )
    
    model_new, metadata, restored_cache = load_matrix_v_model(st_path, model_new)
    print(f"  Metadata Keys: {list(metadata.keys())}")
    if restored_cache:
        print(f"  Restored Cache Size: {len(restored_cache._cache)} entries.")
    
    # --- STEP 4: EXPORT TO ONNX ---
    onnx_path = "matrix_v_model.onnx"
    print(f"\n--- 4. Exporting to ONNX: {onnx_path} ---")
    # MatrixVLinear is substituted with nn.Linear during the export trace
    try:
        export_to_onnx(model, dummy_input, onnx_path)
        print("  ONNX export successful (compatible with ONNX Runtime / TensorRT).")
    except Exception as e:
        print(f"  ONNX export warning: {e} (requires 'onnx' and 'onnxscript' packages)")
    
    # Cleanup
    for p in [st_path, onnx_path]:
        if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    run_distribution_workflow()


