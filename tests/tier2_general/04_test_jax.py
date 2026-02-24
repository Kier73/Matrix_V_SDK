import os
import sys
import jax.numpy as jnp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.extensions.jax_bridge import matrix_v_matmul

def test_jax():
    print("Tier 2: [04] JAX Functional Bridging")
    A = jnp.ones((4, 4))
    B = jnp.ones((4, 4))
    res = matrix_v_matmul(A, B)
    assert res.shape == (4, 4)
    assert res[0,0] == 4.0
    print("[PASS]")

if __name__ == "__main__":
    test_jax()


