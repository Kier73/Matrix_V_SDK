"""
Inverse NTT Law Recovery — Memthematic I/O Gap 2.

Provides modular inverse in the Goldilocks field (P = 2^64 - 2^32 + 1)
via Extended Euclidean Algorithm. Ported from generative_memory's
gmem_trinity.c:mod_inverse().

Core equation:
    parent_b = product_seed * mod_inverse(parent_a) mod P_GOLDILOCKS
"""

from .ntt import P_GOLDILOCKS, NTTMorphism


def extended_gcd(a: int, b: int):
    """Extended Euclidean Algorithm.
    Returns (gcd, x, y) such that a*x + b*y = gcd(a,b).
    """
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y


def mod_inverse(a: int, m: int = P_GOLDILOCKS) -> int:
    """Compute modular inverse: a^{-1} mod m.

    Uses Extended Euclidean Algorithm (identical to gmem_trinity.c).
    Raises ValueError if inverse does not exist (gcd(a,m) != 1).
    """
    a = a % m
    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError(f"Inverse does not exist: gcd({a}, {m}) = {gcd}")
    return x % m


def inverse_product_law(product_seed: int, parent_a_seed: int,
                        p: int = P_GOLDILOCKS) -> int:
    """Recover parent_b_seed from a product and one known parent.

    Given:  product = parent_a * parent_b  mod p
    Solve:  parent_b = product * parent_a^{-1}  mod p

    This is the algebraic decompression operation.
    """
    inv_a = mod_inverse(parent_a_seed, p)
    return NTTMorphism.multiply_mod(product_seed, inv_a)


def verify_law_roundtrip(seed_a: int, seed_b: int,
                         p: int = P_GOLDILOCKS) -> bool:
    """Verify: inverse_product_law(a*b, a) == b."""
    product = NTTMorphism.synthesize_product_law(seed_a, seed_b)
    recovered_b = inverse_product_law(product, seed_a, p)
    return recovered_b == (seed_b % p)


if __name__ == "__main__":
    print("=" * 60)
    print("INVERSE NTT LAW RECOVERY — GAP 2 VERIFICATION")
    print("=" * 60)

    # Test 1: Basic roundtrip
    a, b = 0xDEADBEEF, 0xCAFEBABE
    product = NTTMorphism.synthesize_product_law(a, b)
    recovered = inverse_product_law(product, a)
    print(f"\n  Seed A:     0x{a:X}")
    print(f"  Seed B:     0x{b:X}")
    print(f"  Product:    0x{product:X}")
    print(f"  Recovered:  0x{recovered:X}")
    print(f"  Match:      {recovered == (b % P_GOLDILOCKS)}")

    # Test 2: 1000 random pairs
    import random
    passes = 0
    for _ in range(1000):
        sa = random.randint(1, P_GOLDILOCKS - 1)
        sb = random.randint(1, P_GOLDILOCKS - 1)
        if verify_law_roundtrip(sa, sb):
            passes += 1
    print(f"\n  Roundtrip Test: {passes}/1000 passed")
    print("=" * 60)

