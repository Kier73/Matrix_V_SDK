import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.extensions.torch_bridge import MatrixVLinear

def test_torch():
    print("Tier 2: [03] PyTorch Bridge Basics")
    lin = MatrixVLinear(10, 5)
    x = torch.randn(1, 10)
    out = lin(x)
    assert out.shape == (1, 5)
    print("[PASS]")

if __name__ == "__main__":
    test_torch()


