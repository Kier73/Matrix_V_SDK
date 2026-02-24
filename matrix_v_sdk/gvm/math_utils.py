def hilbert_xy_to_d(n: int, x: int, y: int) -> int:
    """Python implementation of 2D to 1D Hilbert Curve mapping."""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        
        # Rotate/Flip
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            # Swap
            x, y = y, x
        s //= 2
    return d

def vrns_torus_projection(addr: int, seed: int) -> float:
    """Pure Python implementation of VRNS Torus Projection for verification."""
    MODULI = [251, 257, 263, 269, 271, 277, 281, 283]
    x = addr ^ seed
    accumulator = 0.0
    for m in MODULI:
        residue = x % m
        accumulator += float(residue) / float(m)
    return accumulator - int(accumulator)

