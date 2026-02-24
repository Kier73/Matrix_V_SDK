import jax
import jax.numpy as jnp
from jax import custom_vjp
import numpy as np
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega
from .utils import to_list, from_list

# Create a global omega instance for JAX callbacks
_global_omega = MatrixOmega()

def _matrix_v_matmul_impl(A_np, B_np):
    """The raw implementation that the callback will call."""
    res = _global_omega.compute_product(A_np.tolist(), B_np.tolist())
    return np.array(res, dtype=np.float32)

@custom_vjp
def matrix_v_matmul(A, B):
    """
    JAX-compatible matmul using Matrix-V acceleration.
    """
    # Use pure_callback to run the Python/Rust SDK logic inside JAX
    return jax.pure_callback(
        _matrix_v_matmul_impl,
        jax.ShapeDtypeStruct(jnp.matmul(A, B).shape, jnp.float32),
        A, B
    )

def matrix_v_matmul_fwd(A, B):
    return matrix_v_matmul(A, B), (A, B)

def matrix_v_matmul_bwd(res, grad):
    A, B = res
    # dL/dA = grad @ B.T
    # dL/dB = A.T @ grad
    # For simplicity, we use standard JAX matmul for the gradients
    # though we could technically accelerate these too.
    dA = jnp.matmul(grad, B.T)
    dB = jnp.matmul(A.T, grad)
    return dA, dB

matrix_v_matmul.defvjp(matrix_v_matmul_fwd, matrix_v_matmul_bwd)

