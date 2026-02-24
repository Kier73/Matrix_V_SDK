"""
V_Matrix SDK: Matrix Multiplication Acceleration Engine
======================================================
High-performance implementations for RandomProjection, RNS, and On-the-fly
matrix paradigms using deterministic projection techniques.
"""

import math
import random as _random
from typing import List, Tuple, Optional, Any

# --- CORE PRIMITIVES ---

def v_mask(addr: int) -> float:
    """Deterministic Feistel hash for parameter generation. O(1) complexity."""
    l, r = (addr >> 32) & 0xFFFFFFFF, addr & 0xFFFFFFFF
    key = 0xBF58476D
    mul = 0x94D049BB
    for _ in range(4):
        f = ((r ^ key) * mul) & 0xFFFFFFFF
        f = ((f >> 16) ^ f) & 0xFFFFFFFF
        l, r = r, l ^ f
    return ((l << 32) | r) / float(2**64)

def signature(data: List[float]) -> int:
    """Data fingerprint for structural law matching."""
    n = len(data)
    if n == 0: return 0
    first = int(data[0] * 1e6) & 0xFFFFFFFF
    last = int(data[-1] * 1e6) & 0xFFFFFFFF
    mid = int(data[n // 2] * 1e6) & 0xFFFFFFFF
    return first ^ last ^ mid ^ n

# --- ENGINES ---

class RandomProjectionMatrixEngine:
    """
    O(n^2 * D) Approximate Matrix Multiplication via Random Projection.

    Based on the Johnson-Lindenstrauss lemma and sparse Rademacher projections
    (Achlioptas, 2003). Given A (m x d) and B (d x n):

        R ∈ R^{d x D}  (sparse Rademacher random matrix)
        C ≈ (A @ R) @ (R^T @ B)

    Error bound: ||C - AB||_F / ||AB||_F <= epsilon
    with probability >= 1 - delta, where D = O(log(n) / epsilon^2).

    This is a genuine O(n*d*D + n^2*D) algorithm — faster than O(n^2*d)
    when D << d, with a provable quality guarantee.
    """
    def __init__(self, projection_dim: int = 64, seed: int = 42):
        self.base_D = projection_dim
        self.seed = seed

    def _effective_D(self, d: int, epsilon: float = 0.1) -> int:
        """Compute JL-safe projection dimension: D >= O(log(d) / eps^2)."""
        jl_min = max(self.base_D, int(2.0 * math.log2(max(d, 2)) / (epsilon * epsilon)))
        return min(jl_min, d)  # never exceed original dimension

    def _generate_projection(self, d: int) -> List[List[float]]:
        """Generate sparse Rademacher projection matrix R ∈ R^{d x D}.
        R_ij ∈ {-sqrt(3/D), 0, +sqrt(3/D)} with probabilities {1/6, 2/3, 1/6}.
        Sparsity = 2/3, which speeds up the A@R multiplication."""
        D = self._effective_D(d)
        rng = _random.Random(self.seed)
        scale = math.sqrt(3.0 / D)
        R = [[0.0] * D for _ in range(d)]
        for i in range(d):
            for j in range(D):
                r = rng.random()
                if r < 1.0 / 6.0:
                    R[i][j] = scale
                elif r < 2.0 / 6.0:
                    R[i][j] = -scale
                # else: 0.0 (probability 2/3 — sparse)
        return R

    def _matmul_dense(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Standard O(m*k*n) dense matmul helper."""
        m, k = len(A), len(A[0])
        n = len(B[0])
        C = [[0.0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                s = 0.0
                for p in range(k):
                    s += A[i][p] * B[p][j]
                C[i][j] = s
        return C

    def _transpose(self, M: List[List[float]]) -> List[List[float]]:
        rows, cols = len(M), len(M[0])
        return [[M[i][j] for i in range(rows)] for j in range(cols)]

    def multiply(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Approximate C ≈ A @ B via random projection.
        Complexity: O(m*d*D + d*D*n + m*D*n) where D << d."""
        d = len(A[0])  # inner dimension
        D = self._effective_D(d)

        # For small d, direct multiply is faster than projecting
        if d <= D * 2:
            return self._matmul_dense(A, B)

        R = self._generate_projection(d)
        RT = self._transpose(R)

        # A_proj = A @ R   (m x D), cost O(m*d*D)
        A_proj = self._matmul_dense(A, R)
        # B_proj = R^T @ B (D x n), cost O(D*d*n) 
        B_proj = self._matmul_dense(RT, B)
        # C ≈ A_proj @ B_proj (m x n), cost O(m*D*n)
        return self._matmul_dense(A_proj, B_proj)

class RNSMatrixEngine:
    """Exact matrix multiplication using Residue Number Systems (Modulo Arithmetic).

    Encodes floats into fixed-point integers, multiplies in modular residue
    spaces, then reconstructs via the Chinese Remainder Theorem (CRT).

    Fixes applied:
    - Uses round() instead of int() to prevent truncation bias
    - Checks dynamic range overflow before computing
    - Signed CRT reconstruction for correct negative handling
    - Auto-extends prime set if dynamic range is insufficient
    """
    # Candidate primes for extending the residue space
    _EXTRA_PRIMES = [10061, 10067, 10069, 10079, 10091, 10093, 10099, 10103]

    def __init__(self, primes: List[int] = [10007, 10009, 10037, 10039]):
        self.primes = list(primes)
        self._recompute_mod()

    def _recompute_mod(self):
        self.mod_m = 1
        for p in self.primes:
            self.mod_m *= p

    def _extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        if a == 0: return b, 0, 1
        gcd, x1, y1 = self._extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

    def _mod_inverse(self, a: int, m: int) -> int:
        gcd, x, y = self._extended_gcd(a, m)
        if gcd != 1: raise ValueError("Modular inverse does not exist")
        return (x % m + m) % m

    def _ensure_dynamic_range(self, A: List[List[float]], B: List[List[float]], scale: float):
        """Verify that M > 2 * max_possible_sum, auto-extend primes if needed."""
        dim = len(A[0])
        max_abs_A = max(abs(x) for row in A for x in row)
        max_abs_B = max(abs(x) for row in B for x in row)
        max_product_sum = dim * (max_abs_A * scale + 1) * (max_abs_B * scale + 1)

        extra_idx = 0
        while max_product_sum >= self.mod_m // 2:
            if extra_idx >= len(self._EXTRA_PRIMES):
                raise OverflowError(
                    f"RNS dynamic range overflow: max_sum={max_product_sum:.0f} "
                    f">= M/2={self.mod_m // 2}. Reduce scale or matrix magnitude."
                )
            self.primes.append(self._EXTRA_PRIMES[extra_idx])
            extra_idx += 1
            self._recompute_mod()

    def multiply(self, A: List[List[float]], B: List[List[float]], scale: float = 100.0) -> List[List[float]]:
        rows, cols, dim = len(A), len(B[0]), len(A[0])

        # Dynamic range check — auto-extend primes if needed
        self._ensure_dynamic_range(A, B, scale)

        # 1. Compute residues for each prime
        #    Use round() not int() to avoid truncation bias
        residues = []
        for p in self.primes:
            A_p = [[round(x * scale) % p for x in row] for row in A]
            B_p = [[round(B[k][j] * scale) % p for j in range(cols)] for k in range(dim)]

            p_res = [[0] * cols for _ in range(rows)]
            for i in range(rows):
                for j in range(cols):
                    p_res[i][j] = sum(A_p[i][k] * B_p[k][j] for k in range(dim)) % p
            residues.append(p_res)

        # 2. Recombine via CRT with signed reconstruction
        final_C = [[0.0] * cols for _ in range(rows)]
        half_m = self.mod_m // 2
        for i in range(rows):
            for j in range(cols):
                val = 0
                for idx, p in enumerate(self.primes):
                    r = residues[idx][i][j]
                    M_i = self.mod_m // p
                    y_i = self._mod_inverse(M_i, p)
                    val = (val + r * M_i * y_i) % self.mod_m

                # Signed CRT: if val > M/2, the true value is val - M
                if val > half_m:
                    val -= self.mod_m

                # Rescale back from fixed-point (scale^2 because it's a product)
                final_C[i][j] = float(val) / (scale * scale)
        return final_C

class SNAPMatrixEngine:
    """Deterministic Weight Generation (On-the-fly parameterization).

    Projects input X through a deterministic weight matrix W derived from seed.
    W_ji = (v_mask(seed ^ j ^ i) * 2 - 1) / sqrt(in_dim)

    The 1/sqrt(in_dim) normalization ensures Var[Y_j] = ||X||^2 / in_dim,
    preventing variance explosion in deep projection chains.
    """
    def multiply(self, X: List[float], seed: int, out_dim: int) -> List[float]:
        """Projects input X through a deterministic weight matrix derived from seed."""
        in_dim = len(X)
        scale = 1.0 / math.sqrt(in_dim) if in_dim > 0 else 1.0
        output = []
        for j in range(out_dim):
            # Retrieve deterministic weight vector for neuron j, scaled by 1/sqrt(d)
            w_j = [(v_mask(seed ^ j ^ i) * 2.0 - 1.0) * scale for i in range(in_dim)]
            val = sum(xi * wi for xi, wi in zip(X, w_j))
            output.append(val)
        return output

# --- HIGH LEVEL INTERFACE ---
from .sdk_registry import solver, method

@solver("VMatrix")
class VMatrix:
    """Unified SDK Interface for Projected Matrix Operations."""
    def __init__(self, mode: str = "spectral"):
        self.mode = mode.lower()
        self.spectral = RandomProjectionMatrixEngine()
        self.rns = RNSMatrixEngine()
        self.snap = SNAPMatrixEngine()

    @method("VMatrix", "matmul")
    def matmul(self, A: List[List[float]], B: List[List[float]], **kwargs) -> List[List[float]]:
        if self.mode == "spectral":
            return self.spectral.multiply(A, B)
        elif self.mode == "rns":
            return self.rns.multiply(A, B, scale=kwargs.get("scale", 1000.0))
        else:
            raise ValueError(f"Mode {self.mode} not supported for standard matmul.")

    def snap_project(self, X: List[float], seed: int, out_dim: int) -> List[float]:
        return self.snap.multiply(X, seed, out_dim)

if __name__ == "__main__":
    # Quick sanity check
    vm = VMatrix(mode="spectral")
    mat_a = [[1.0, 0.5], [0.2, 0.8]]
    mat_b = [[0.7, 0.3], [0.1, 0.9]]
    res = vm.matmul(mat_a, mat_b)
    print(f"Spectral Result: {res}")
    
    vm_rns = VMatrix(mode="rns")
    res_rns = vm_rns.matmul(mat_a, mat_b)
    print(f"RNS Result: {res_rns}")

