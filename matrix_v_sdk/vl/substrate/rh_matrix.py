import math
import random
from typing import List, Any, Dict
from .sdk_registry import solver, method
from .prime_matrix import PrimeMatrix
from .x_matrix import XMatrix, HdcManifold

# --- CORE ALGORITHMIC GROUNDING ---

def is_prime(n, k=5):
    """Miller-Rabin primality test for extreme-scale coordinates."""
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0: return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else: return False
    return True

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def pollard_rho(n):
    """Pollard's rho with Brent improvement for non-trivial factorization.
    Retries with different random constants instead of recursive calls."""
    if n % 2 == 0: return 2
    if is_prime(n): return n
    for _ in range(20):
        x = random.randint(2, n - 1)
        y, c, g = x, random.randint(1, n - 1), 1
        while g == 1:
            x = (pow(x, 2, n) + c) % n
            y = (pow(y, 2, n) + c) % n
            y = (pow(y, 2, n) + c) % n
            g = gcd(abs(x - y), n)
            if g == n: break
        if g != n:
            return g
    raise ArithmeticError(f"Pollard-rho failed to factor {n}")

def get_mobius(n: int, timeout_iters: int = 100) -> int:
    """Analytical Mobius resolution using the P-Series prime logic."""
    if n == 1: return 1
    temp, prime_factors, iters = n, set(), 0
    # Small prime pre-screening
    for d in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        if temp % d == 0:
            prime_factors.add(d)
            temp //= d
            if temp % d == 0: return 0
        if temp == 1: break
    # Factorization loop
    while temp > 1 and iters < timeout_iters:
        if is_prime(temp):
            if temp in prime_factors: return 0
            prime_factors.add(temp)
            temp = 1
            break
        factor = pollard_rho(temp)
        if is_prime(factor):
            if factor in prime_factors: return 0
            prime_factors.add(factor)
            temp //= factor
        iters += 1
    if temp > 1:
        raise ArithmeticError(f"Failed to fully factor {n}: unfactored remainder {temp}")
    return -1 if len(prime_factors) % 2 == 1 else 1

# --- RECTIFIED SOLVERS (Using V, X, P Foundations) ---

@solver("MobiusMatrix")
class MobiusMatrix:
    """
    Generation 4.5: Analytical Mobius Inversion.
    Foundation: Built on P-Series (PrimeMatrix) and X-Series (Descriptor Binding).
    """
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)
        # Structural Base (P-Series)
        self.p_series = PrimeMatrix(rows, cols)
        # Semantic Descriptor (X-Series) for structural identity tracking
        self.x_series = XMatrix(rows, cols, seed=0x71346)

    @method("MobiusMatrix", "get_element")
    def get_element(self, r: int, c: int) -> int:
        # Utilize PrimeMatrix (P-Series) for the sparsity mask (divisibility)
        if self.p_series.get_element(r, c) == 0:
            return 0
        # Resolve Mobius coefficient for the non-zero entry
        return get_mobius((c + 1) // (r + 1))

@solver("RedhefferMatrix")
class RedhefferMatrix:
    """
    Redheffer Matrix Formulation.
    Foundation: P-Series (Divisibility) composed with Unit-Column (V-Series logic).
    """
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)
        self.p_series = PrimeMatrix(rows, cols)

    @method("RedhefferMatrix", "get_element")
    def get_element(self, r: int, c: int) -> int:
        col_val = c + 1
        # Redheffer identity: 1 if i divides j OR j == 1
        if col_val == 1: return 1
        return self.p_series.get_element(r, c)

    def mertens_sample(self, n: int) -> int:
        """Grounding check for small-scale verification."""
        return sum(get_mobius(k) for k in range(1, n + 1))

    @staticmethod
    def mertens_sieve(n: int) -> List[int]:
        """Compute M(1) through M(n) in O(n log log n) via linear sieve.

        Returns a list where result[k] = M(k) = sum_{i=1}^{k} mu(i).
        Much faster than per-element Pollard-rho for bulk computation.
        """
        mu = [0] * (n + 1)
        mu[1] = 1
        is_p = [True] * (n + 1)
        primes = []
        for i in range(2, n + 1):
            if is_p[i]:
                primes.append(i)
                mu[i] = -1  # i is prime => squarefree with 1 factor
            for p in primes:
                if i * p > n:
                    break
                is_p[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0  # p^2 divides i*p => not squarefree
                    break
                mu[i * p] = -mu[i]  # one more prime factor
        # Prefix sum
        M = [0] * (n + 1)
        for i in range(1, n + 1):
            M[i] = M[i - 1] + mu[i]
        return M

    def structural_mertens_sample(self, start: int, sample_size: int = 100) -> Dict[str, Any]:
        """Statistical profiling of Mobius distributions at large scales."""
        results = []
        for i in range(sample_size):
            try:
                results.append(get_mobius(start + i))
            except ArithmeticError:
                results.append(None)  # Mark as unfactored
        valid = [r for r in results if r is not None]
        counts = {
            "mu_0": valid.count(0),
            "mu_1": valid.count(1),
            "mu_neg1": valid.count(-1),
            "unfactored": results.count(None)
        }
        local_sum = sum(r for r in valid if r in [-1, 0, 1])
        total_valid = len(valid) if valid else 1
        return {
            "start": start,
            "sample_size": sample_size,
            "distribution": counts,
            "local_sum": local_sum,
            "density_sq_free": (counts["mu_1"] + counts["mu_neg1"]) / total_valid
        }

