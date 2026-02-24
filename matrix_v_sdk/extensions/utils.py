import numpy as np
try:
    import torch
except ImportError:
    torch = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

def to_list(data):
    """Converts various array types to nested Python lists for SDK consumption."""
    if isinstance(data, list):
        return data
    if isinstance(data, np.ndarray):
        return data.tolist()
    if torch is not None and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().tolist()
    if jnp is not None and isinstance(data, (jnp.ndarray, jax.Array)):
        return np.array(data).tolist()
    return list(data)

def from_list(data, target_type='numpy'):
    """Converts nested lists back to the target framework type."""
    if target_type == 'numpy':
        return np.array(data)
    if target_type == 'torch' and torch is not None:
        return torch.tensor(data)
    if target_type == 'jax' and jnp is not None:
        return jnp.array(data)
    return data

