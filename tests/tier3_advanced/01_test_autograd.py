import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.extensions.torch_bridge import MatrixVLinear

def test_autograd():
    print("Tier 3: [01] Training with MatrixVLinear Autograd")
    model = nn.Sequential(
        MatrixVLinear(10, 20),
        nn.ReLU(),
        MatrixVLinear(20, 1)
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    initial_loss = nn.MSELoss()(model(x), y).item()
    for _ in range(10):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
    
    final_loss = nn.MSELoss()(model(x), y).item()
    print(f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}")
    assert final_loss < initial_loss
    print("[PASS]")

if __name__ == "__main__":
    test_autograd()


