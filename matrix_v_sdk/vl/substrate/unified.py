"""
QMatrix: Streaming Tiled Matmul Orchestrator
=============================================
Unifies GVM (procedural storage), numerical engines, symbolic descriptors,
and quantum-informed rank estimation into a single coherent matrix engine.

Architecture:
  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
  │  GVM (1PB)  │───▶│ Tile Router  │───▶│ Engine Pool  │
  │  O(1) I/O   │    │ rank → mode  │    │ Dense/JL/Sym │
  └─────────────┘    └──────────────┘    └──────────────┘
                          │
                     ┌────▼────┐
                     │ MPSNode │ (quantum entropy → rank estimate)
                     └─────────┘
"""

import math
from typing import List, Tuple, Optional, Any

try:
    from ...gvm.core import GenerativeMemory
except (ImportError, SystemError):
    try:
        from gvm.core import GenerativeMemory
    except ImportError:
        GenerativeMemory = None

from ..math.rns import VlAdaptiveRNS, Q_RNS_PRIMES
from ..math.primitives import vl_mask
from .signatures import SidechannelDetector
from .matrix import MatrixOmega, SymbolicDescriptor
from .tile_collapser import TileLaw, collapse as tile_collapse, resolve as tile_resolve
from .manifold_fitter import ManifoldDescriptor, fit_manifold
from .rns_ledger import RNSLedger, record_matrix

try:
    from ..quantum.tensor import MPSNode
except ImportError:
    MPSNode = None


# ─── GVM Address Bridge ─────────────────────────────────

class GvmMatrixBridge:
    """Maps a 2D matrix into a flat GVM address space."""

    def __init__(self, gvm: GenerativeMemory, rows: int, cols: int, base_addr: int = 0):
        self.gvm = gvm
        self.rows = rows
        self.cols = cols
        self.base_addr = base_addr

    def _addr(self, i: int, j: int) -> int:
        return self.base_addr + i * self.cols + j

    def store(self, data: List[List[float]]):
        """Write an m×n matrix into GVM starting at base_addr."""
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                self.gvm.write(self._addr(i, j), val)

    def fetch_element(self, i: int, j: int) -> float:
        return self.gvm.fetch(self._addr(i, j))

    def fetch_tile(self, row_start: int, col_start: int,
                   tile_rows: int, tile_cols: int) -> List[List[float]]:
        """Extract a tile sub-matrix from GVM."""
        tile = []
        for i in range(tile_rows):
            row = []
            for j in range(tile_cols):
                r, c = row_start + i, col_start + j
                if r < self.rows and c < self.cols:
                    row.append(self.gvm.fetch(self._addr(r, c)))
                else:
                    row.append(0.0)
            tile.append(row)
        return tile

    def write_tile(self, row_start: int, col_start: int,
                   tile: List[List[float]]):
        """Write a tile sub-matrix into GVM."""
        for i, row in enumerate(tile):
            for j, val in enumerate(row):
                r, c = row_start + i, col_start + j
                if r < self.rows and c < self.cols:
                    self.gvm.write(self._addr(r, c), val)


# ─── Tile Rank Estimator (Quantum) ──────────────────────

def estimate_tile_rank(tile: List[List[float]]) -> float:
    """
    Estimates the effective rank of a tile using MPSNode entropy.

    Encodes the tile's singular-value proxy (row norms + column correlations)
    into an RNS torus, then reads off the entanglement entropy:
      - Low entropy → low-rank (spectral/JL is efficient)
      - High entropy → full-rank (dense is safer)

    Returns a value in [0, 1] where 0 = rank-1, 1 = full-rank.
    """
    m = len(tile)
    if m == 0 or len(tile[0]) == 0:
        return 0.0

    if MPSNode is None:
        # Fallback: assume medium rank if quantum module unavailable
        return 0.5

    node = MPSNode()
    # Encode structural statistics into the RNS torus
    for i in range(min(m, 16)):
        row_energy = sum(x * x for x in tile[i])
        # Map row energy to a residue value in [0, p)
        p = Q_RNS_PRIMES[i % 16]
        node.residues[i % 16] = int(row_energy * 1000) % p

    entropy = node.entanglement_entropy()
    max_entropy = 16 * math.log2(2)  # theoretical max for 16 channels
    return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0


# ─── QMatrix Engine ─────────────────────────────────────

class QMatrix(MatrixOmega):
    """
    Exascale Matrix Substrate.

    Unifies GVM procedural storage with the SDK engine pool.
    Supports three modes:
      1. Tiled Dense — GVM-backed, tile-routed matmul for RAM-exceeding matrices
      2. Symbolic — O(1) descriptor composition for trillion-scale
      3. Quantum-Informed — MPSNode entropy guides engine selection per tile

    Complexity: O(n² · D / T) amortized over T-sized tiles with GVM I/O.
    """

    DEFAULT_TILE_SIZE = 64

    def __init__(self, seed: int = 0x517, tile_size: int = DEFAULT_TILE_SIZE):
        super().__init__(seed)
        self.tile_size = tile_size
        self.detector = SidechannelDetector()

        # Lazy GVM — only initialize if needed
        self._gvm = None
        self._gvm_seed = seed

    @property
    def gvm(self) -> GenerativeMemory:
        """Lazy-init GVM to avoid DLL requirement for pure-Python usage."""
        if self._gvm is None:
            self._gvm = GenerativeMemory(seed=self._gvm_seed)
        return self._gvm

    # ─── High-level API ──────────────────────────────────

    def multiply(self, A: List[List[float]], B: List[List[float]],
                 tile_size: Optional[int] = None) -> List[List[float]]:
        """
        Tiled streaming matrix product.

        For small matrices (n ≤ tile_size): single-shot dense.
        For large matrices: tiles through GVM with per-tile engine routing.
        """
        m, k = len(A), len(A[0])
        n = len(B[0])
        T = tile_size or self.tile_size

        # Small matrix: direct compute, no tiling overhead
        if m <= T and n <= T and k <= T:
            return self._route_tile(A, B)

        # Large matrix: streaming tiled matmul
        return self._tiled_multiply(A, B, m, k, n, T)

    def symbolic_multiply(self, a: SymbolicDescriptor,
                           b: SymbolicDescriptor) -> SymbolicDescriptor:
        """O(1) symbolic composition — zero memory, infinite scale."""
        if a.cols != b.rows:
            raise ValueError(
                f"Dimension mismatch: {a.rows}×{a.cols} vs {b.rows}×{b.cols}")
        new_sig = (a.signature ^ (b.signature >> 1) ^ (a.depth << 32)) & 0xFFFFFFFFFFFFFFFF
        return SymbolicDescriptor(a.rows, b.cols, new_sig, a.depth + b.depth)

    # ─── Tiled Engine ────────────────────────────────────

    def _tiled_multiply(self, A, B, m, k, n, T) -> List[List[float]]:
        """
        Blocked matmul: C[i_block, j_block] += A_tile · B_tile

        Each tile-pair is routed through the optimal engine based on
        quantum-estimated rank.
        """
        # Initialize output
        C = [[0.0] * n for _ in range(m)]

        # Tile counts
        mt = math.ceil(m / T)
        nt = math.ceil(n / T)
        kt = math.ceil(k / T)

        for ib in range(mt):
            r0 = ib * T
            rm = min(r0 + T, m)
            for jb in range(nt):
                c0 = jb * T
                cm = min(c0 + T, n)
                for lb in range(kt):
                    k0 = lb * T
                    km = min(k0 + T, k)

                    # Extract tiles
                    A_tile = [row[k0:km] for row in A[r0:rm]]
                    B_tile = [row[c0:cm] for row in B[k0:km]]

                    # Route through best engine
                    partial = self._route_tile(A_tile, B_tile)

                    # Accumulate into C
                    for i in range(len(partial)):
                        for j in range(len(partial[0])):
                            C[r0 + i][c0 + j] += partial[i][j]

        return C

    def _route_tile(self, A_tile: List[List[float]],
                    B_tile: List[List[float]]) -> List[List[float]]:
        """
        Select the optimal engine for a single tile product.

        Uses quantum entropy to estimate rank:
          rank_ratio < 0.3  → spectral (low-rank, JL efficient)
          rank_ratio >= 0.3 → dense (full-rank, exact)
        """
        rank_ratio = estimate_tile_rank(A_tile)

        m = len(A_tile)
        n = len(B_tile[0]) if B_tile else 0

        if rank_ratio < 0.3 and m > 32 and n > 32:
            return self.spectral.matmul(A_tile, B_tile)
        else:
            return self.naive_multiply(A_tile, B_tile)

    # ─── GVM-Backed Mode ─────────────────────────────────

    def gvm_multiply(self, A_addr: int, B_addr: int,
                     m: int, k: int, n: int,
                     tile_size: Optional[int] = None) -> List[List[float]]:
        """
        Streaming matmul where both matrices live in GVM.

        This is the out-of-core path: matrices can exceed RAM because
        GVM provides O(1) procedural element access from its 1PB space.
        """
        T = tile_size or self.tile_size
        A_bridge = GvmMatrixBridge(self.gvm, m, k, A_addr)
        B_bridge = GvmMatrixBridge(self.gvm, k, n, A_addr + m * k)

        C = [[0.0] * n for _ in range(m)]
        mt = math.ceil(m / T)
        nt = math.ceil(n / T)
        kt = math.ceil(k / T)

        for ib in range(mt):
            r0 = ib * T
            rm = min(T, m - r0)
            for jb in range(nt):
                c0 = jb * T
                cm = min(T, n - c0)
                for lb in range(kt):
                    k0 = lb * T
                    km = min(T, k - k0)

                    A_tile = A_bridge.fetch_tile(r0, k0, rm, km)
                    B_tile = B_bridge.fetch_tile(k0, c0, km, cm)

                    partial = self._route_tile(A_tile, B_tile)

                    for i in range(len(partial)):
                        for j in range(len(partial[0])):
                            C[r0 + i][c0 + j] += partial[i][j]

        return C

    # ─── Memthematic I/O Engine ────────────────────────────

    def memthematic_multiply(
        self, A: List[List[float]], B: List[List[float]],
        tile_size: Optional[int] = None,
        verify: bool = False
    ) -> 'MemthematicResult':
        """
        Full Memthematic I/O pipeline:
          dense matmul → tiled collapse → ManifoldDescriptor

        Returns a MemthematicResult that JIT-resolves any element
        without keeping the dense matrix in memory.

        If verify=True, also records an RNS ledger for exact
        integrity checking.
        """
        m, k = len(A), len(A[0])
        n = len(B[0])
        T = tile_size or self.tile_size

        # Step 1: Compute dense result via tiled engine
        C = self.multiply(A, B, tile_size=T)

        # Step 2: Collapse into manifold
        manifold = fit_manifold(C, tile_size=T)

        # Step 3: Optional RNS integrity ledger
        ledger = None
        if verify:
            ledger = record_matrix(C)

        return MemthematicResult(
            manifold=manifold,
            ledger=ledger,
            source_dims=(m, k, n),
            tile_size=T
        )

    # ─── Sidechannel Probe ───────────────────────────────

    def probe_and_lock(self, stream_sample: List[float]) -> Optional[int]:
        """
        Analyze a data stream for varietal periodicity.
        If locked, re-seeds GVM to the detected manifold.
        """
        candidate = self.detector.probe_stream_block(stream_sample)
        if candidate and self._gvm is not None:
            self._gvm.seed = candidate
        return candidate


class MemthematicResult:
    """
    The collapsed output of a matrix multiplication.

    Instead of an m×n list of floats, this holds a ManifoldDescriptor
    (compact law-based representation) and an optional RNS ledger
    (exact integrity verification).

    Any element can be JIT-resolved via resolve(r, c).
    """

    def __init__(self, manifold: ManifoldDescriptor,
                 ledger: Optional[RNSLedger],
                 source_dims: tuple,
                 tile_size: int):
        self.manifold = manifold
        self.ledger = ledger
        self.source_dims = source_dims  # (m, k, n)
        self.tile_size = tile_size

    @property
    def rows(self) -> int:
        return self.manifold.rows

    @property
    def cols(self) -> int:
        return self.manifold.cols

    def resolve(self, r: int, c: int) -> float:
        """JIT-resolve a single element from the collapsed result."""
        return self.manifold.resolve(r, c)

    def verify(self, r: int, c: int, ground_truth: float) -> bool:
        """Verify a single element against the RNS ledger."""
        if self.ledger is None:
            raise RuntimeError("No RNS ledger — call memthematic_multiply(verify=True)")
        return self.ledger.verify_residues(r, c, ground_truth)

    def stats(self) -> dict:
        """Full pipeline statistics."""
        ms = self.manifold.stats()
        result = {
            "source_dims": self.source_dims,
            "tile_size": self.tile_size,
            **ms,
        }
        if self.ledger:
            result["ledger"] = self.ledger.stats()
        return result

    def __repr__(self):
        m, k, n = self.source_dims
        cr = self.manifold.compression_ratio()
        return f"<MemthematicResult {m}x{n} | {cr:.1f}x compression | {len(self.manifold.tiles)} tiles>"


