import math
import random

def get_mobius(n):
    """Analytical Mobius resolution via prime parity."""
    if n == 1: return 1
    factors = {}
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            factors[d] = 0
            while temp % d == 0:
                factors[d] += 1
                temp //= d
            if factors[d] > 1: return 0 # Not square-free
        d += 1
    if temp > 1: factors[temp] = 1
    return -1 if len(factors) % 2 == 1 else 1

def det_redheffer(n):
    """The determinant of an n x n Redheffer matrix is the Mertens function M(n)."""
    return sum(get_mobius(k) for k in range(1, n + 1))

if __name__ == "__main__":
    N = 100
    m_val = det_redheffer(N)
    print(f"det(Redheffer_{N}) = M({N}) = {m_val}")

