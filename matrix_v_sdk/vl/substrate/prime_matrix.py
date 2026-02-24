import math
import random as _random
from typing import List, Any
from .sdk_registry import solver, method


def _is_prime(n: int, k: int = 10) -> bool:
    """Miller-Rabin primality test for large-scale factorization."""
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0: return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for _ in range(k):
        a = _random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else: return False
    return True


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _pollard_rho(n: int) -> int:
    """Pollard-rho with Brent improvement for non-trivial factor finding."""
    if n % 2 == 0: return 2
    if _is_prime(n): return n
    for _ in range(20):
        x = _random.randint(2, n - 1)
        y, c, g = x, _random.randint(1, n - 1), 1
        while g == 1:
            x = (pow(x, 2, n) + c) % n
            y = (pow(y, 2, n) + c) % n
            y = (pow(y, 2, n) + c) % n
            g = _gcd(abs(x - y), n)
            if g == n: break
        if g != n:
            return g
    raise ArithmeticError(f"Pollard-rho failed to factor {n}")


def _get_prime_factors(n: int) -> dict:
    """Factor n completely using trial division + Pollard-rho.
    Works reliably for n up to ~10^18."""
    factors = {}
    # Trial division for small primes
    for d in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
    # Extended trial division up to sqrt for moderate n
    if n > 1 and n < 10_000_000_000:
        d = 53
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 2
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
    else:
        # Pollard-rho for large composites
        while n > 1:
            if _is_prime(n):
                factors[n] = factors.get(n, 0) + 1
                break
            f = _pollard_rho(n)
            while n % f == 0:
                factors[f] = factors.get(f, 0) + 1
                n //= f
    return factors


@solver("PrimeMatrix")
class PrimeMatrix:
    """
    Generation 4: Analytical Divisor Matrix.
    Represents a divisor matrix where P[i, j] = 1 if (i+1) divides (j+1), else 0.
    Symbolic multiplication: P^m counts m-step divisor chains via prime factorization.
    """
    def __init__(self, rows: int, cols: int, depth: int = 1):
        self.rows = rows
        self.cols = cols
        self.depth = depth
        self.shape = (rows, cols)

    def get_element(self, r: int, c: int) -> Any:
        """
        Mathematically grounded resolution for arbitrary depth (m).
        Returns the number of divisor chains k_0 | k_1 | ... | k_m
        where k_0 = r+1 and k_m = c+1.

        Formula: Product of C(a_p + m-1, m-1) over prime factors p^a_p of X = (c+1)/(r+1).
        """
        row_val = r + 1
        col_val = c + 1

        if col_val % row_val != 0:
            return 0

        if self.depth == 1:
            return 1

        X = col_val // row_val
        m = self.depth

        # Fast path: X is a power of 2
        if X > 0 and (X & (X - 1)) == 0:
            a = X.bit_length() - 1
            return math.comb(a + m - 1, m - 1)

        # General case: full factorization via trial division + Pollard-rho
        factors = _get_prime_factors(X)
        res = 1
        for p, a in factors.items():
            res *= math.comb(a + m - 1, m - 1)
        return res

    def multiply(self, other: 'PrimeMatrix') -> 'PrimeMatrix':
        """
        Symbolic Multiplication.
        The product of two divisor matrices represents the 'Composite Divisor Matrix'.
        """
        if self.cols != other.rows:
            raise ValueError("Dimension mismatch")
        
        # The product of two divisor matrices P * P has elements:
        # (P^2)_{ij} = count of k such that i|k and k|j.
        # This is exactly the number of divisors of (j/i).
        return PrimeMatrix(self.rows, other.cols, depth=self.depth + other.depth)

if __name__ == "__main__":
    # Quick sanity check for 10x10 divisor matrix
    pm = PrimeMatrix(10, 10)
    print("10x10 Divisor Matrix (1-indexed logic):")
    for r in range(10):
        row = [int(pm.get_element(r, c)) for c in range(10)]
        print(row)

