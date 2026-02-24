import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.extensions.hf_bridge import MatrixVAttention

def test_trans_100():
    print("Tier 4: [06] 100-Layer Unrolled Attention Chain")
    # Simulation of a very deep network
    x = torch.randn(1, 10, 64)
    for i in range(100):
        attn = MatrixVAttention(embed_dim=64, num_heads=2)
        x = attn(x) + x # Residual
    print(f"Output Norm after 100 layers: {x.norm().item()}")
    assert x.shape == (1, 10, 64)
    print("[PASS]")

if __name__ == "__main__":
    test_trans_100()


