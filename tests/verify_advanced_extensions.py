import torch
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add SDK paths
sys.path.append(os.path.abspath(os.curdir))
from matrix_v_sdk.extensions.jax_bridge import matrix_v_matmul
from matrix_v_sdk.extensions.hf_bridge import MatrixVAttention

def test_jax_bridge():
    print("\n--- [VERIFICATION] JAX Extension (Custom VJP) ---")
    A = jnp.ones((8, 8), dtype=jnp.float32)
    B = jnp.ones((8, 8), dtype=jnp.float32)
    
    # Forward pass
    output = matrix_v_matmul(A, B)
    print(f"JAX Forward pass result (8x8 all ones): {output[0,0]} (expected 8.0)")
    
    # Backward pass
    grad_fn = jax.grad(lambda x, y: matrix_v_matmul(x, y).sum())
    dA = grad_fn(A, B)
    print(f"JAX Gradient dA[0,0]: {dA[0,0]} (expected 8.0)")
    
    if abs(output[0,0] - 8.0) < 1e-5 and abs(dA[0,0] - 8.0) < 1e-5:
        print("  [v] JAX BRIDGE FUNCTIONAL")
    else:
        print("  [x] JAX BRIDGE FAILED")

def test_transformer_attention():
    print("\n--- [VERIFICATION] Transformer Attention Bridge ---")
    embed_dim = 64
    num_heads = 8
    seq_len = 16
    batch_size = 1
    
    attn = MatrixVAttention(embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    output = attn(x)
    print(f"Attention output shape: {output.shape}")
    
    if output.shape == (batch_size, seq_len, embed_dim):
        print("  [v] TRANSFORMER ATTENTION FUNCTIONAL")
    else:
        print("  [x] TRANSFORMER ATTENTION FAILED")

if __name__ == "__main__":
    try:
        test_jax_bridge()
        test_transformer_attention()
    except Exception as e:
        import traceback
        traceback.print_exc()


