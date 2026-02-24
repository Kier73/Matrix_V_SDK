"""
SafeTensors Bridge
------------------
Save/load Matrix-V models using the safetensors format
(HuggingFace standard for model weights).

Embeds StrategyCache state in safetensors metadata dict,
so learned adaptive strategies persist across save/load cycles.
"""
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

import json
import os


def save_matrix_v_model(model, path, strategy_cache=None, extra_metadata=None):
    """
    Save a Matrix-V model's weights to safetensors format.

    Args:
        model: nn.Module with MatrixVLinear layers
        path: Output .safetensors file path
        strategy_cache: Optional StrategyCache to embed in metadata
        extra_metadata: Optional dict of additional metadata

    The StrategyCache is serialized as JSON in the metadata dict,
    keyed under 'matrix_v_strategy_cache'.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for safetensors bridge")
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors package required (pip install safetensors)")

    # Collect all parameter tensors
    tensors = {}
    for name, param in model.named_parameters():
        tensors[name] = param.data

    # Build metadata
    metadata = extra_metadata or {}
    metadata['matrix_v_sdk_version'] = '1.0'

    # Serialize StrategyCache if provided
    if strategy_cache is not None:
        cache_data = strategy_cache.to_dict()
        metadata['matrix_v_strategy_cache'] = json.dumps(cache_data)

    # Collect MatrixVLinear layer info
    from .torch_bridge import MatrixVLinear
    layer_info = {}
    for name, module in model.named_modules():
        if isinstance(module, MatrixVLinear):
            layer_info[name] = {
                'in_features': module.in_features,
                'out_features': module.out_features,
                'has_bias': module.bias is not None,
            }
    if layer_info:
        metadata['matrix_v_layers'] = json.dumps(layer_info)

    # safetensors metadata must be str -> str
    str_metadata = {k: str(v) for k, v in metadata.items()}

    save_file(tensors, path, metadata=str_metadata)
    return path


def load_matrix_v_model(path, model):
    """
    Load weights from a safetensors file into a Matrix-V model.

    Args:
        path: Path to .safetensors file
        model: nn.Module to load weights into

    Returns:
        (model, metadata_dict) — model with loaded weights, plus
        any embedded metadata including StrategyCache state.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for safetensors bridge")
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors package required (pip install safetensors)")

    tensors = load_file(path)

    # Load parameters into model
    state_dict = model.state_dict()
    for name, tensor in tensors.items():
        if name in state_dict:
            state_dict[name] = tensor
    model.load_state_dict(state_dict)

    # Extract metadata
    metadata = {}
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            metadata = dict(f.metadata()) if f.metadata() else {}
    except Exception:
        pass

    # Restore StrategyCache if present
    strategy_cache = None
    if 'matrix_v_strategy_cache' in metadata:
        from matrix_v_sdk.vl.substrate.matrix import StrategyCache
        cache_data = json.loads(metadata['matrix_v_strategy_cache'])
        strategy_cache = StrategyCache.from_dict(cache_data)

        # Inject into MatrixVLinear layers
        from .torch_bridge import MatrixVLinear
        for name, module in model.named_modules():
            if isinstance(module, MatrixVLinear):
                module.omega.strategy_cache = strategy_cache

    return model, metadata, strategy_cache

