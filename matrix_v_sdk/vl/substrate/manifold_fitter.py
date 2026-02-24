"""
Manifold Fitter — Memthematic I/O Gap 3A.

Piecewise Manifold Fitting: decomposes a dense matrix into tiles,
collapses each tile into a TileLaw, and stores the result as a
"Manifold Descriptor" — a compact representation from which any
element can be JIT-resolved.

Architecture:
  Dense Matrix C (m×n)
       ↓ Tile(T)
  [(tile_0, bounds_0), (tile_1, bounds_1), ...]
       ↓ Collapse()
  [(TileLaw_0, bounds_0), (TileLaw_1, bounds_1), ...]
       = ManifoldDescriptor
       ↓ resolve(r, c)
  C[r][c]  (approximate or exact depending on tile classification)
"""

import math
from typing import List, Tuple, Optional, NamedTuple

from .tile_collapser import TileLaw, collapse, resolve


class TileEntry(NamedTuple):
    """A collapsed tile with its position in the parent matrix."""
    r_start: int
    c_start: int
    law: TileLaw


class ManifoldDescriptor:
    """A piecewise manifold: the 'folded' representation of a dense matrix.

    Memory footprint: O(ceil(m/T) * ceil(n/T)) tile laws
    instead of O(m*n) floats.
    """

    def __init__(self, rows: int, cols: int, tile_size: int,
                 tiles: List[TileEntry]):
        self.rows = rows
        self.cols = cols
        self.tile_size = tile_size
        self.tiles = tiles

        # Build lookup grid for O(1) tile access
        self._mt = math.ceil(rows / tile_size)
        self._nt = math.ceil(cols / tile_size)
        self._grid = {}
        for entry in tiles:
            tb = entry.r_start // tile_size
            tc = entry.c_start // tile_size
            self._grid[(tb, tc)] = entry

    def resolve(self, r: int, c: int) -> float:
        """JIT-resolve a single element from the manifold."""
        tb = r // self.tile_size
        tc = c // self.tile_size
        entry = self._grid.get((tb, tc))
        if entry is None:
            return 0.0  # Unmapped region
        local_r = r - entry.r_start
        local_c = c - entry.c_start
        return resolve(entry.law, local_r, local_c)

    def memory_footprint_bytes(self) -> int:
        """Estimated memory of the descriptor vs the original matrix."""
        # Each TileLaw: ~48 bytes (rule str + sig + p1 + p2 + rows + cols)
        return len(self.tiles) * 48

    def compression_ratio(self) -> float:
        """Compression ratio vs dense storage (8 bytes per float)."""
        dense_bytes = self.rows * self.cols * 8
        if dense_bytes == 0:
            return 1.0
        return dense_bytes / self.memory_footprint_bytes()

    def stats(self) -> dict:
        """Classification statistics across all tiles."""
        counts = {"zero": 0, "constant": 0, "linear": 0, "complex": 0}
        for entry in self.tiles:
            counts[entry.law.rule] = counts.get(entry.law.rule, 0) + 1
        return {
            "total_tiles": len(self.tiles),
            "classification": counts,
            "compression_ratio": self.compression_ratio(),
            "descriptor_bytes": self.memory_footprint_bytes(),
            "dense_bytes": self.rows * self.cols * 8,
        }


def fit_manifold(matrix: List[List[float]], tile_size: int = 32,
                 epsilon: float = 1e-6) -> ManifoldDescriptor:
    """Fit a piecewise manifold to a dense matrix.

    Decomposes the matrix into T×T tiles, classifies each via ARO,
    and returns a ManifoldDescriptor.
    """
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    T = tile_size
    tiles = []

    for r0 in range(0, rows, T):
        rm = min(r0 + T, rows)
        for c0 in range(0, cols, T):
            cm = min(c0 + T, cols)

            # Extract tile
            tile = [row[c0:cm] for row in matrix[r0:rm]]

            # Collapse
            law = collapse(tile, epsilon)

            tiles.append(TileEntry(r_start=r0, c_start=c0, law=law))

    return ManifoldDescriptor(rows, cols, T, tiles)


def verify_manifold_parity(matrix: List[List[float]],
                           manifold: ManifoldDescriptor,
                           epsilon: float = 1e-4) -> Tuple[bool, float, int]:
    """Verify parity between original matrix and manifold resolution.

    Returns (passed, max_error, error_count).
    """
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    max_err = 0.0
    error_count = 0

    for r in range(rows):
        for c in range(cols):
            original = matrix[r][c]
            resolved = manifold.resolve(r, c)
            err = abs(original - resolved)
            max_err = max(max_err, err)
            if err >= epsilon:
                error_count += 1

    return error_count == 0, max_err, error_count


if __name__ == "__main__":
    import random
    print("=" * 60)
    print("MANIFOLD FITTER — GAP 3A VERIFICATION")
    print("=" * 60)

    # Test 1: Uniform constant matrix
    const_matrix = [[5.0] * 64 for _ in range(64)]
    mf = fit_manifold(const_matrix, tile_size=16)
    print(f"\n  Constant 64×64 Matrix:")
    print(f"    Stats: {mf.stats()}")
    ok, err, ec = verify_manifold_parity(const_matrix, mf)
    print(f"    Parity: {'PASS' if ok else 'FAIL'} (max_err={err:.2e})")

    # Test 2: Linear ramp matrix
    linear_matrix = [[0.1 * (r * 64 + c) for c in range(64)] for r in range(64)]
    mf = fit_manifold(linear_matrix, tile_size=16)
    print(f"\n  Linear 64×64 Matrix:")
    print(f"    Stats: {mf.stats()}")
    ok, err, ec = verify_manifold_parity(linear_matrix, mf)
    print(f"    Parity: {'PASS' if ok else 'FAIL'} (max_err={err:.2e})")

    # Test 3: Mixed (constant blocks + random blocks)
    random.seed(42)
    mixed = [[0.0] * 64 for _ in range(64)]
    for r in range(64):
        for c in range(64):
            if r < 32 and c < 32:
                mixed[r][c] = 7.0  # Constant quadrant
            elif r < 32:
                mixed[r][c] = 0.0  # Zero quadrant
            else:
                mixed[r][c] = random.gauss(0, 1)  # Random quadrant
    mf = fit_manifold(mixed, tile_size=16)
    print(f"\n  Mixed 64×64 Matrix:")
    print(f"    Stats: {mf.stats()}")

    print("=" * 60)

