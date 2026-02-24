try:
    from numba import njit, prange
except ImportError:
    # Fallback if numba is not installed
    def njit(*args, **kwargs):
        def wrapper(f): return f
        return wrapper
    prange = range

import numpy as np
from matrix_v_sdk.vl.substrate.acceleration import P_SeriesEngine

@njit(parallel=True)
def numba_p_series_resolved(m, n, p_val):
    """
    JIT-optimized P-Series resolution for tandem operation.
    """
    res = np.zeros((m, n), dtype=np.float64)
    for i in prange(m):
        for j in range(n):
            # Inline logic from P_SeriesEngine for maximum speed
            # Since Engine logic might be complex, we define a JIT-compatible subset
            val = (i + 1) * (j + 1)
            if val % p_val == 0:
                res[i, j] = float(val) / p_val
            else:
                res[i, j] = 0.0
    return res

class NumbaBridge:
    """
    Provides access points for Numba-accelerated categorical kernels.
    """
    @staticmethod
    def resolve_p_lattice(m, n, p_val):
        """
        Uses @njit parallel kernel to resolve a P-Series lattice.
        """
        return numba_p_series_resolved(m, n, p_val)

    @staticmethod
    def accelerate_custom_kernel(kernel_fn):
        """
        Wraps a custom categorical kernel with njit for the user.
        """
        return njit(parallel=True)(kernel_fn)

