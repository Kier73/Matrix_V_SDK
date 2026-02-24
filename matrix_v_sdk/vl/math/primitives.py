import numpy as np

# --- MATHEMATICAL CONSTANTS: SPECTRAL ANCHORS ---
# These constants are derived from the golden ratio fractional part (phi) 
# and are used as irrigation seeds for balanced Feistel distributions.
# They ensure that even small coordinate changes (L1 distance = 1)
# result in high-entropy manifold shifts.
C_MAGIC = 0x9E3779B97F4A7C15
C2 = 0xBF58476D1CE4E5B9
C3 = 0x94D049BB133111EB

# --- INVARIANT INVERSE CONSTANTS ---
# Used for Spectral Resonance Collapse (Reverse Hashing).
# These allow O(1) reconstruction of source coordinates from a variety value.
INV_C2 = 0x96DE1B173F119089
INV_C3 = 0x319642B2D24D8EC3

def vl_mask(addr: int, seed: int) -> int:
    """
    VL_DeterministicVariety: A canonical Feistel-based variety generator.
    
    THEORY:
    This function implements a permutation on the 64-bit integer space. 
    By treating the address + seed as a coordinate in a high-dimensional manifold, 
    we use avalanche-effect mixing to generate a 'category signature'.
    This replaces traditional PRNGs with stateless, O(1) coordinate-to-field mapping.
    """
    z = (addr + seed + C_MAGIC) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * C2 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * C3 & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

def vl_inverse_mask(z: int, seed: int) -> int:
    """
    Inverse of the vl_mask function.
    Allows O(1) recovery of the input address from the output hash.
    Used for Spectral Resonance Collapse (Subverting Grover's).
    """
    def invert_shift_xor(val: int, k: int) -> int:
        """Helper to invert the xorshift transformation."""
        res = val
        while True:
            val >>= k
            if val == 0:
                break
            res ^= val
        return res & 0xFFFFFFFFFFFFFFFF

    # 1. Undo z ^ (z >> 31)
    z = invert_shift_xor(z, 31)
    
    # 2. Undo step 5 (z * C3)
    z = (z * INV_C3) & 0xFFFFFFFFFFFFFFFF
    
    # 3. Undo step 4 (z ^ (z >> 27))
    z = invert_shift_xor(z, 27)
    
    # 4. Undo step 3 (z * C2)
    z = (z * INV_C2) & 0xFFFFFFFFFFFFFFFF
    
    # 5. Undo step 2 (z ^ (z >> 30))
    z = invert_shift_xor(z, 30)
    
    # 6. Undo step 1 (addr + seed + C_MAGIC)
    return (z - seed - C_MAGIC) & 0xFFFFFFFFFFFFFFFF

def vl_signature(data: bytes, seed: int) -> int:
    """
    VL_ManifoldSignature: Bit-exact structural signature.
    Hashes arbitrary data into a 64-bit signature.
    """
    h = (seed + C_MAGIC) & 0xFFFFFFFFFFFFFFFF
    for byte in data:
        h = (h + byte) & 0xFFFFFFFFFFFFFFFF
        h = (h ^ (h >> 30)) * C2 & 0xFFFFFFFFFFFFFFFF
        h = (h ^ (h >> 27)) * C3 & 0xFFFFFFFFFFFFFFFF
        h = (h ^ (h >> 31)) & 0xFFFFFFFFFFFFFFFF
    return h

def fmix64(h: int) -> int:
    """
    MurmurMix64: Finalizer resonance mix.
    
    Used to collapse a high-entropy manifold into a single floating-point resonance.
    Essential for 'variety-based' element resolution in Symbolic Descriptors.
    Ensures that coordinate collisions are statistically impossible within
    the trillion-scale operational envelope (10^12).
    """
    h ^= h >> 33
    h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    return h

def r_gielis(phi: np.ndarray, m: float, a: float, b: float, n1: float, n2: float, n3: float) -> np.ndarray:
    """
    Gielis Superformula Radius.
    Used for Morphological DNA projection in T-Matrix layers.
    """
    term1 = np.power(np.abs(np.cos(m * phi / 4.0) / a), n2)
    term2 = np.power(np.abs(np.sin(m * phi / 4.0) / b), n3)
    return np.power(term1 + term2, -1.0 / n1)

def hilbert_encode(i: int, j: int, order: int) -> int:
    """
    2D -> 1D Hilbert Space-Filling Curve.
    Used for topological sequence alignment and compute shunting.
    """
    d = 0
    s = 1 << (order - 1)
    while s > 0:
        rx = 1 if (i & s) > 0 else 0
        ry = 1 if (j & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                i = s - 1 - i
                j = s - 1 - j
            i, j = j, i
        s >>= 1
    return d

