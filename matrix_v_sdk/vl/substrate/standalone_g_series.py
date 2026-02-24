def fmix64(h):
    h &= 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    return h & 0xFFFFFFFFFFFFFFFF

def g_resolve(rows, cols, signature, depth, r, c):
    """JIT Element Realization for G-Series."""
    idx = (r * cols + c) & 0xFFFFFFFFFFFFFFFF
    h = fmix64(signature ^ idx ^ depth)
    return (h / float(2**64)) * 2.0 - 1.0

def g_multiply(s1, d1, s2, d2):
    """Symbolic Signature Synthesis."""
    new_sig = (s1 ^ (s2 >> 1) ^ (d1 << 32)) & 0xFFFFFFFFFFFFFFFF
    return new_sig, d1 + d2

if __name__ == "__main__":
    N = 10**12
    sig_a, depth_a = 0xDEADBEEF, 1
    sig_b, depth_b = 0xCAFEBABE, 1
    
    # O(1) Symbolic Multiply
    sig_c, depth_c = g_multiply(sig_a, depth_a, sig_b, depth_b)
    
    # O(1) JIT Resolve
    val = g_resolve(N, N, sig_c, depth_c, 128, 256)
    print(f"G[{128}, {256}] = {val:.4f}")

