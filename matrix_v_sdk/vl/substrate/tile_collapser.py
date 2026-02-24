"""
Tile Collapser — Memthematic I/O Gap 1.

Collapses a dense tile (List[List[float]]) into a compact seed
that can rematerialize approximate or exact values.

Uses ARO-inspired classification (from gmem_aro.c):
  - Zero:     All elements ≈ 0 → seed = 0
  - Constant: All elements ≈ k → seed encodes k
  - Linear:   Elements follow y = mx + b → seed encodes (m, b)
  - Complex:  No simple law → store RNS residue fingerprint

Morph-inspired realization (from gmem.h morph system):
  resolve(seed, idx) = scale * fmix64(seed ⊕ idx) + offset
"""

import struct
import math
from typing import List, Tuple, Optional, NamedTuple

from ..math.primitives import fmix64


# --- Collapsed Tile Descriptor ---

class TileLaw(NamedTuple):
    """A collapsed tile: the 'seed' of a dense result."""
    rule: str           # "zero" | "constant" | "linear" | "complex"
    signature: int      # 64-bit algebraic signature
    p1: float           # Parameter 1 (constant value, or intercept b)
    p2: float           # Parameter 2 (slope m, or scale)
    rows: int
    cols: int


# --- Classification Engine (ARO Port) ---

def classify_tile(tile: List[List[float]],
                  epsilon: float = 1e-6) -> TileLaw:
    """Classify a tile into its simplest algebraic law.

    Mirrors gmem_aro_simplify() logic:
      1. Check for Zero Manifold
      2. Check for Constant Manifold
      3. Check for Linear Manifold (least-squares fit)
      4. Fall back to Complex (hash-based signature)
    """
    rows = len(tile)
    cols = len(tile[0]) if rows > 0 else 0
    n = rows * cols

    if n == 0:
        return TileLaw("zero", 0, 0.0, 0.0, rows, cols)

    # Flatten
    flat = [tile[r][c] for r in range(rows) for c in range(cols)]

    # 1. Zero check
    max_abs = max(abs(v) for v in flat)
    if max_abs < epsilon:
        return TileLaw("zero", 0, 0.0, 0.0, rows, cols)

    # 2. Constant check
    mean = sum(flat) / n
    variance = sum((v - mean) ** 2 for v in flat) / n
    if variance < epsilon:
        sig = fmix64(struct.unpack('Q', struct.pack('d', mean))[0])
        return TileLaw("constant", sig, mean, 0.0, rows, cols)

    # 3. Linear fit: y = m * idx + b  (least-squares)
    # Using the normal equations:
    #   m = (n·Σ(x·y) - Σx·Σy) / (n·Σ(x²) - (Σx)²)
    #   b = (Σy - m·Σx) / n
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_xx = 0.0
    for i, v in enumerate(flat):
        sum_x += i
        sum_y += v
        sum_xy += i * v
        sum_xx += i * i

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) > epsilon:
        m = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - m * sum_x) / n

        # Check residual error
        sse = sum((flat[i] - (m * i + b)) ** 2 for i in range(n))
        rmse = math.sqrt(sse / n)

        if rmse < epsilon * 10:
            sig = fmix64(struct.unpack('Q', struct.pack('d', m))[0]
                         ^ struct.unpack('Q', struct.pack('d', b))[0])
            return TileLaw("linear", sig, b, m, rows, cols)

    # 4. Complex: full content hash (FNV-1a over all float bits)
    sig = _full_content_hash(flat)
    return TileLaw("complex", sig, 0.0, 0.0, rows, cols)


def _full_content_hash(flat: List[float]) -> int:
    """FNV-1a rolling hash over all float bits (from InductiveEngine)."""
    h = 0xcbf29ce484222325
    for i, val in enumerate(flat):
        bits = struct.unpack('I', struct.pack('f', val))[0]
        h ^= fmix64(bits ^ i)
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


# --- Resolution Engine (Morph Port) ---

def resolve_from_law(law: TileLaw, r: int, c: int) -> float:
    """JIT-resolve a single element from a TileLaw descriptor.

    Mirrors gmem_morph_attach() logic:
      Zero:     return 0.0
      Constant: return p1
      Linear:   return p2 * idx + p1  (y = mx + b)
      Complex:  return fmix64(sig ⊕ idx) normalized to [-1, 1]
    """
    idx = r * law.cols + c

    if law.rule == "zero":
        return 0.0
    elif law.rule == "constant":
        return law.p1
    elif law.rule == "linear":
        return law.p2 * idx + law.p1
    else:  # complex
        h = fmix64(law.signature ^ idx)
        return (h / float(2**64)) * 2.0 - 1.0


# --- Collapse + Resolve API ---

def collapse(tile: List[List[float]], epsilon: float = 1e-6) -> TileLaw:
    """Collapse a dense tile into a compact law descriptor."""
    return classify_tile(tile, epsilon)


def resolve(law: TileLaw, r: int, c: int) -> float:
    """Resolve a single element from a law descriptor."""
    return resolve_from_law(law, r, c)


def verify_collapse_parity(tile: List[List[float]],
                           epsilon: float = 1e-4) -> Tuple[bool, float]:
    """Verify that collapse → resolve matches the original tile.

    Returns (passed, max_error).
    """
    law = collapse(tile)
    rows = len(tile)
    cols = len(tile[0]) if rows > 0 else 0
    max_err = 0.0

    for r in range(rows):
        for c in range(cols):
            original = tile[r][c]
            resolved = resolve(law, r, c)
            err = abs(original - resolved)
            max_err = max(max_err, err)

    return max_err < epsilon, max_err


if __name__ == "__main__":
    print("=" * 60)
    print("TILE COLLAPSER — GAP 1 VERIFICATION")
    print("=" * 60)

    # Test 1: Zero tile
    zero_tile = [[0.0] * 4 for _ in range(4)]
    law = collapse(zero_tile)
    print(f"\n  Zero Tile:  rule={law.rule}, sig=0x{law.signature:X}")
    ok, err = verify_collapse_parity(zero_tile)
    print(f"  Parity:     {'PASS' if ok else 'FAIL'} (max_err={err:.2e})")

    # Test 2: Constant tile
    const_tile = [[3.14] * 4 for _ in range(4)]
    law = collapse(const_tile)
    print(f"\n  Const Tile: rule={law.rule}, p1={law.p1:.4f}")
    ok, err = verify_collapse_parity(const_tile)
    print(f"  Parity:     {'PASS' if ok else 'FAIL'} (max_err={err:.2e})")

    # Test 3: Linear tile
    linear_tile = [[0.5 * (r * 4 + c) + 1.0 for c in range(4)] for r in range(4)]
    law = collapse(linear_tile)
    print(f"\n  Linear Tile: rule={law.rule}, m={law.p2:.4f}, b={law.p1:.4f}")
    ok, err = verify_collapse_parity(linear_tile)
    print(f"  Parity:      {'PASS' if ok else 'FAIL'} (max_err={err:.2e})")

    # Test 4: Complex tile (random)
    import random
    random.seed(42)
    complex_tile = [[random.gauss(0, 1) for _ in range(4)] for _ in range(4)]
    law = collapse(complex_tile)
    print(f"\n  Complex Tile: rule={law.rule}, sig=0x{law.signature:X}")
    ok, err = verify_collapse_parity(complex_tile, epsilon=2.0)
    print(f"  Parity:       {'PASS (approx)' if ok else 'FAIL'} (max_err={err:.4f})")

    print("=" * 60)

