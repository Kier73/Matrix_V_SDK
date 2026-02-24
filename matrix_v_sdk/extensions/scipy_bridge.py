"""
SciPy Sparse Bridge
-------------------
Enables CSR/CSC sparse matrices to flow through the adaptive classifier.

Key insight: sparse format gives us O(1) sparsity via nnz/(m*n),
avoiding the sampling overhead in MatrixFeatureVector.from_matrices().
"""
try:
    import scipy.sparse as sp
except ImportError:
    sp = None

import numpy as np
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega, MatrixFeatureVector


def _sparse_to_list(mat):
    """Convert any scipy sparse matrix to List[List[float]]."""
    if sp is None:
        raise ImportError("scipy is required for sparse bridge")
    return mat.toarray().tolist()


def _list_to_sparse(data, format='csr'):
    """Convert List[List[float]] back to scipy sparse."""
    if sp is None:
        raise ImportError("scipy is required for sparse bridge")
    arr = np.array(data)
    if format == 'csr':
        return sp.csr_matrix(arr)
    elif format == 'csc':
        return sp.csc_matrix(arr)
    elif format == 'coo':
        return sp.coo_matrix(arr)
    return sp.csr_matrix(arr)


def sparse_feature_vector(A_sparse, B) -> MatrixFeatureVector:
    """
    Build a MatrixFeatureVector from a sparse matrix in O(1).
    
    Instead of sampling elements (which requires dense access),
    we read structural metadata directly from the sparse format:
      sparsity = 1 - nnz / (m * k)
    """
    m, k = A_sparse.shape
    if sp.issparse(B):
        n = B.shape[1]
    elif hasattr(B, 'shape'):
        n = B.shape[1]
    else:
        n = len(B[0]) if B else 0

    fv = MatrixFeatureVector(m=m, k=k, n=n)
    fv.flop_cost = 2 * m * k * n
    fv.is_square = (m == k == n)
    fv.is_rectangular_bottleneck = (k > 5 * m and k > 5 * n)

    # O(1) sparsity from sparse format metadata
    total_elements = m * k
    fv.sparsity = 1.0 - (A_sparse.nnz / total_elements) if total_elements > 0 else 0.0

    # Row variance from CSR row pointer (O(m))
    if sp.issparse(A_sparse) and hasattr(A_sparse, 'indptr'):
        csr = A_sparse.tocsr()
        row_nnz = np.diff(csr.indptr).astype(np.float64)
        fv.row_variance = float(np.var(row_nnz))

    return fv


class SparseMatrixV:
    """
    Sparse-aware wrapper around MatrixOmega.

    Usage:
        smv = SparseMatrixV()
        C_sparse = smv.multiply(A_csr, B_csr)

    The adaptive classifier uses sparse metadata for routing:
    - Very sparse (>90%) -> spectral engine (low effective rank)
    - Moderate sparse -> standard dense path
    - Structured sparse -> inductive engine if tiling detected
    """
    def __init__(self, omega=None):
        self.omega = omega or MatrixOmega()

    def multiply(self, A, B, output_format='csr'):
        """
        Multiply two matrices (sparse or dense) through the adaptive system.
        
        Returns result in the requested sparse format.
        """
        # Convert sparse inputs to lists for the SDK
        A_list = _sparse_to_list(A) if sp and sp.issparse(A) else (A.tolist() if hasattr(A, 'tolist') else A)
        B_list = _sparse_to_list(B) if sp and sp.issparse(B) else (B.tolist() if hasattr(B, 'tolist') else B)

        result = self.omega.compute_product(A_list, B_list)

        # Return in requested format
        if output_format in ('csr', 'csc', 'coo'):
            return _list_to_sparse(result, format=output_format)
        elif output_format == 'dense':
            return np.array(result)
        return result

    def feature_vector(self, A, B=None):
        """Get the feature vector for a sparse matrix (O(1) sparsity)."""
        if sp and sp.issparse(A):
            return sparse_feature_vector(A, B)
        return MatrixFeatureVector.from_matrices(
            A.tolist() if hasattr(A, 'tolist') else A,
            B.tolist() if hasattr(B, 'tolist') else B
        )

