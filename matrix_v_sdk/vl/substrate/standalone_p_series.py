import math
import random

def is_prime(n, k=10):
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0: return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1; d //= 2
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
    while b: a, b = b, a % b
    return a

def pollard_rho(n):
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
        if g != n: return g
    return n

def get_prime_factors(n):
    factors = {}
    for d in [2, 3, 5, 7, 11, 13, 17, 19]:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
    while n > 1:
        if is_prime(n):
            factors[n] = factors.get(n, 0) + 1
            break
        f = pollard_rho(n)
        while n % f == 0:
            factors[f] = factors.get(f, 0) + 1
            n //= f
    return factors

def p_series_resolve(i, j, m):
    """(P^m)_ij = Product_{p|X} C(ap + m - 1, m - 1) where X = j/i"""
    if j % i != 0: return 0
    X = j // i
    if m == 1: return 1
    factors = get_prime_factors(X)
    res = 1
    for p, a in factors.items():
        res *= math.comb(a + m - 1, m - 1)
    return res

if __name__ == "__main__":
    # Test at Billion scale
    i, j, m = 1, 248832, 3  # 248832 = 2^10 * 3^5
    val = p_series_resolve(i, j, m)
    print(f"P^{m}[{i}, {j}] = {val} (Expected: 1386)")

