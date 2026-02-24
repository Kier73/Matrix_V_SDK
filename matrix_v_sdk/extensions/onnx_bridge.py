"""
ONNX Export Bridge
------------------
Enables exporting MatrixVLinear models to ONNX format for deployment
to inference engines (TensorRT, ONNX Runtime, OpenVINO).

Strategy: MatrixVFunction maps to standard ONNX MatMul + Add,
since the SDK's custom engines are CPU-side optimizations that
don't need to be embedded in the graph — the weights are the same.
"""
try:
    import torch
    import torch.nn as nn
    import torch.onnx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

import os
import numpy as np


def _make_exportable_model(model):
    """
    Create an ONNX-exportable copy of a model by replacing MatrixVLinear
    with standard nn.Linear (same weights, standard ONNX ops).
    
    Avoids deepcopy (fails on ctypes objects in MatrixOmega) by
    building replacements in-place on a state_dict level.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for ONNX export")

    from .torch_bridge import MatrixVLinear
    
    # Build a mapping of modules to replace
    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, MatrixVLinear):
            linear = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None
            )
            linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                linear.bias.data = module.bias.data.clone()
            replacements[name] = linear

    if not replacements:
        return model  # No MatrixVLinear layers, return as-is

    # Apply replacements by walking the module hierarchy
    # For nn.Sequential, replace by index; for named children, replace by attr
    for name, replacement in replacements.items():
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            if hasattr(parent, part):
                parent = getattr(parent, part)
            else:
                parent = parent[int(part)]
        
        last = parts[-1]
        if isinstance(parent, nn.Sequential):
            parent[int(last)] = replacement
        else:
            setattr(parent, last, replacement)
    
    return model


def export_to_onnx(model, dummy_input, path, opset_version=14, 
                   input_names=None, output_names=None, dynamic_axes=None):
    """
    Export a Matrix-V model to ONNX format.

    Replaces MatrixVLinear layers with standard nn.Linear for ONNX
    compatibility. The SDK acceleration is a runtime optimization —
    the learned weights are identical.

    Args:
        model: nn.Module containing MatrixVLinear layers
        dummy_input: Example input tensor for tracing
        path: Output .onnx file path
        opset_version: ONNX opset (default 14)
        input_names: Names for input nodes
        output_names: Names for output nodes
        dynamic_axes: Dict of dynamic axes for variable-length inputs

    Returns:
        Path to exported ONNX file
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for ONNX export")
    
    exportable = _make_exportable_model(model)
    
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    
    torch.onnx.export(
        exportable,
        dummy_input,
        path,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes or {},
        do_constant_folding=True,
    )
    
    return path


def validate_onnx(path):
    """Validate that an exported ONNX model is well-formed."""
    if not ONNX_AVAILABLE:
        raise ImportError("onnx package required for validation")
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return True


def compare_outputs(model, onnx_path, test_input, atol=1e-5):
    """
    Compare PyTorch model output vs ONNX Runtime inference.
    
    Returns:
        dict with max_error, mean_error, and match (bool)
    """
    if not ORT_AVAILABLE:
        raise ImportError("onnxruntime required for comparison")
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for comparison")
    
    # PyTorch forward
    model.eval()
    with torch.no_grad():
        pt_output = model(test_input).numpy()
    
    # ONNX Runtime forward
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    ort_output = session.run(None, {input_name: test_input.numpy()})[0]
    
    max_err = float(np.max(np.abs(pt_output - ort_output)))
    mean_err = float(np.mean(np.abs(pt_output - ort_output)))
    
    return {
        "max_error": max_err,
        "mean_error": mean_err,
        "match": max_err < atol,
    }

