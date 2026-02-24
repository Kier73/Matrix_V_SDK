import os
import sys
import jax
import jax.numpy as jnp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.extensions.jax_bridge import matrix_v_matmul

def test_jax_grad():
    print("Tier 3: [03] Differentiating Through Categorical Kernels")
    @jax.grad
    def loss_fn(A):
        B = jnp.eye(4)
        return matrix_v_matmul(A, B).sum()
    
    A = jnp.ones((4, 4))
    grad = loss_fn(A)
    assert grad.shape == (4, 4)
    print(f"Grad Sum: {grad.sum()}")
    print("[PASS]")

if __name__ == "__main__":
    test_jax_grad()


