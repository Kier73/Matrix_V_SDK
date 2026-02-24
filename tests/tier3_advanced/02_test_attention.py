import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.extensions.hf_bridge import MatrixVAttention

def test_attention():
    print("Tier 3: [02] Transformer Self-Attention Forward Pass")
    attn = MatrixVAttention(embed_dim=128, num_heads=4)
    x = torch.randn(1, 32, 128)
    out = attn(x)
    assert out.shape == (1, 32, 128)
    print("[PASS]")

if __name__ == "__main__":
    test_attention()


