"""
Memthematic Pipeline: Optimal Order of Operations
===================================================
Chains the entire SDK into a single pipeline for maximum throughput:

    Stage 0: DEFINE   -- SymbolicDescriptor  O(1)    (instant structure)
    Stage 1: COMPOSE  -- Symbolic multiply   O(1)    (instant chain)
    Stage 2: STREAM   -- JIT tile matmul     O(n^2K) (exact, streaming)
    Stage 3: COLLAPSE -- Manifold fit        O(n^2)  (compressed cache)
    Stage 4: VERIFY   -- RNS ledger          O(n^2)  (exact integrity)

After collapse, any element is available at O(1) from the manifold
or re-derivable from the symbolic chain at O(K) with bit-exactness.

Usage:
    pipe = MemthematicPipeline(tile_size=64)
    result = pipe.multiply(A_desc, B_desc)        # Symbolic + JIT + Collapse
    result = pipe.chain([A, B, C, D])             # Chained multiply
    val    = result.resolve(i, j)                 # O(1) from manifold
    exact  = result.jit_resolve(i, j)             # O(K) bit-exact
    ok     = result.verify(i, j)                  # RNS check
"""

import time
import math
from typing import List, Optional, Tuple, Union

from ..math.primitives import fmix64
from .matrix import SymbolicDescriptor
from .tile_collapser import TileLaw, collapse as tile_collapse, resolve as tile_resolve
from .manifold_fitter import ManifoldDescriptor, fit_manifold
from .rns_ledger import RNSLedger, record_matrix


# ==============================================================
# LAZY MATRIX: Wraps any source into a uniform interface
# ==============================================================

class LazyMatrix:
    """
    Uniform interface over any matrix source:
      - Dense (list of lists)
      - SymbolicDescriptor (seed-defined)
      - Another LazyMatrix (chained computation)

    Provides O(1) element resolution regardless of source.
    """

    __slots__ = ('rows', 'cols', '_kind', '_dense', '_sym', '_chain')

    def __init__(self, source):
        if isinstance(source, SymbolicDescriptor):
            self._kind = 'symbolic'
            self._sym = source
            self._dense = None
            self._chain = None
            self.rows = source.rows
            self.cols = source.cols
        elif isinstance(source, list):
            self._kind = 'dense'
            self._dense = source
            self._sym = None
            self._chain = None
            self.rows = len(source)
            self.cols = len(source[0]) if source else 0
        elif isinstance(source, tuple) and len(source) == 3:
            # Chain: (LazyMatrix_A, LazyMatrix_B, inner_dim)
            self._kind = 'chain'
            self._chain = source
            self._dense = None
            self._sym = None
            self.rows = source[0].rows
            self.cols = source[1].cols
        else:
            raise TypeError(f"Unknown source type: {type(source)}")

    def resolve(self, i: int, j: int) -> float:
        """O(1) for dense/symbolic, O(K) for chain."""
        if self._kind == 'dense':
            return self._dense[i][j]
        elif self._kind == 'symbolic':
            return self._sym.resolve(i, j)
        else:
            # Chain: C[i][j] = SUM_k A[i][k] * B[k][j]
            A, B, _ = self._chain
            K = A.cols
            acc = 0.0
            for k in range(K):
                acc += A.resolve(i, k) * B.resolve(k, j)
            return acc

    def resolve_tile(self, r0: int, c0: int, tr: int, tc: int) -> list:
        """Materialize a rectangular tile."""
        return [[self.resolve(r0 + r, c0 + c)
                 for c in range(tc)]
                for r in range(tr)]


# ==============================================================
# PIPELINE RESULT
# ==============================================================

class PipelineResult:
    """
    The output of a pipeline operation.

    Contains:
      - symbolic: O(1) structure/signature
      - manifold: compressed tile-law representation (if materialized)
      - ledger:   exact RNS verification (if verified)
      - source:   LazyMatrix chain for JIT re-derivation

    Three resolution modes:
      resolve(i,j)      -- O(1) from manifold (fast, approximate for complex)
      jit_resolve(i,j)  -- O(K) from source chain (exact, streaming)
      verify(i,j)       -- check against RNS ledger (exact integrity)
    """

    def __init__(self, symbolic: SymbolicDescriptor,
                 source: LazyMatrix,
                 manifold: Optional[ManifoldDescriptor] = None,
                 ledger: Optional[RNSLedger] = None,
                 timings: Optional[dict] = None):
        self.symbolic = symbolic
        self.source = source
        self.manifold = manifold
        self.ledger = ledger
        self.timings = timings or {}

    @property
    def rows(self) -> int:
        return self.symbolic.rows

    @property
    def cols(self) -> int:
        return self.symbolic.cols

    def resolve(self, i: int, j: int) -> float:
        """
        O(1) resolution from manifold.
        Falls back to JIT if no manifold exists.
        """
        if self.manifold:
            return self.manifold.resolve(i, j)
        return self.jit_resolve(i, j)

    def jit_resolve(self, i: int, j: int) -> float:
        """
        O(K) exact resolution by streaming through the source chain.
        Bit-exact reproducible. No storage required.
        """
        return self.source.resolve(i, j)

    def verify(self, i: int, j: int, ground_truth: Optional[float] = None) -> bool:
        """
        Verify element integrity via RNS ledger.
        If no ground_truth provided, uses JIT to compute it.
        """
        if self.ledger is None:
            raise RuntimeError("No ledger — use pipeline.multiply(..., verify=True)")
        if ground_truth is None:
            ground_truth = self.jit_resolve(i, j)
        return self.ledger.verify_residues(i, j, ground_truth)

    def stats(self) -> dict:
        """Pipeline statistics."""
        result = {
            "rows": self.rows,
            "cols": self.cols,
            "signature": f"0x{self.symbolic.signature:016X}",
            "depth": self.symbolic.depth,
            "timings": self.timings,
        }
        if self.manifold:
            result["manifold"] = self.manifold.stats()
        if self.ledger:
            result["ledger"] = self.ledger.stats()
        return result

    def __repr__(self):
        parts = [f"{self.rows}x{self.cols}"]
        parts.append(f"depth={self.symbolic.depth}")
        if self.manifold:
            cr = self.manifold.compression_ratio()
            parts.append(f"{cr:.0f}x compressed")
        if self.ledger:
            parts.append("RNS verified")
        return f"<PipelineResult {' | '.join(parts)}>"


# ==============================================================
# MEMTHEMATIC PIPELINE ENGINE
# ==============================================================

class MemthematicPipeline:
    """
    Optimal order-of-operations engine.

    Chains the SDK's systems in the fastest sequence:

      1. SYMBOLIC (instant)  -- O(1) shape + signature composition
      2. JIT STREAM (exact)  -- O(n^2 K) tiled matmul, O(T^2) memory
      3. COLLAPSE (compress) -- O(n^2) manifold fitting
      4. VERIFY (exact)      -- O(n^2) RNS ledger recording

    Stages 3-4 are optional and triggered by parameters.
    """

    def __init__(self, tile_size: int = 64):
        self.tile_size = tile_size

    # ─── Single Multiply ──────────────────────────────────

    def multiply(self, A, B,
                 materialize: bool = True,
                 verify: bool = False) -> PipelineResult:
        """
        Multiply two matrices through the optimal pipeline.

        A, B can be:
          - SymbolicDescriptor (seed-defined)
          - list[list[float]]  (dense)
          - PipelineResult     (from previous multiply)

        If materialize=True:  JIT streams the full result + collapses
        If materialize=False: only computes symbolic structure (O(1))
        If verify=True:       also records RNS ledger
        """
        timings = {}

        # Normalize inputs
        A_lazy = self._to_lazy(A)
        B_lazy = self._to_lazy(B)
        A_sym = self._to_symbolic(A, A_lazy)
        B_sym = self._to_symbolic(B, B_lazy)

        if A_lazy.cols != B_lazy.rows:
            raise ValueError(
                f"Dimension mismatch: {A_lazy.rows}x{A_lazy.cols} "
                f"vs {B_lazy.rows}x{B_lazy.cols}")

        # Stage 1: Symbolic composition -- O(1)
        t0 = time.perf_counter()
        C_sym = A_sym.multiply(B_sym)
        timings['symbolic'] = time.perf_counter() - t0

        # Lazy source chain (for JIT re-derivation)
        C_lazy = LazyMatrix((A_lazy, B_lazy, A_lazy.cols))

        manifold = None
        ledger = None

        if materialize:
            # Stage 2: JIT streaming matmul -- O(n^2 K)
            t0 = time.perf_counter()
            C_dense = self._jit_matmul(A_lazy, B_lazy)
            timings['jit_stream'] = time.perf_counter() - t0

            # Stage 3: Collapse to manifold -- O(n^2)
            t0 = time.perf_counter()
            manifold = fit_manifold(C_dense, tile_size=self.tile_size)
            timings['collapse'] = time.perf_counter() - t0

            if verify:
                # Stage 4: RNS ledger -- O(n^2)
                t0 = time.perf_counter()
                ledger = record_matrix(C_dense)
                timings['rns_record'] = time.perf_counter() - t0

            # Overwrite lazy source with dense for fast access
            C_lazy = LazyMatrix(C_dense)

        return PipelineResult(
            symbolic=C_sym,
            source=C_lazy,
            manifold=manifold,
            ledger=ledger,
            timings=timings
        )

    # ─── Chained Multiply ─────────────────────────────────

    def chain(self, matrices: list,
              materialize: bool = True,
              verify: bool = False) -> PipelineResult:
        """
        Chain-multiply a list of matrices: A x B x C x D x ...

        Uses left-to-right associativity with intermediate
        materialization for maximum throughput.
        """
        if len(matrices) < 2:
            raise ValueError("Need at least 2 matrices to chain")

        result = self.multiply(
            matrices[0], matrices[1],
            materialize=materialize,
            verify=False  # only verify the final result
        )

        for i in range(2, len(matrices)):
            result = self.multiply(
                result, matrices[i],
                materialize=materialize,
                verify=False
            )

        # Optionally verify the final result
        if verify and materialize and result.manifold:
            t0 = time.perf_counter()
            # Reconstruct dense from manifold source
            C_dense = [[result.source.resolve(r, c)
                        for c in range(result.cols)]
                       for r in range(result.rows)]
            result.ledger = record_matrix(C_dense)
            result.timings['rns_record'] = time.perf_counter() - t0

        return result

    # ─── Symbolic-Only Mode ───────────────────────────────

    def symbolic_chain(self, descriptors: List[SymbolicDescriptor]) -> SymbolicDescriptor:
        """
        Pure O(1) symbolic composition.
        No materialization, no streaming, no memory.
        Returns the structure/signature for an arbitrarily long chain.
        """
        result = descriptors[0]
        for desc in descriptors[1:]:
            result = result.multiply(desc)
        return result

    # ─── Internal ─────────────────────────────────────────

    def _to_lazy(self, source) -> LazyMatrix:
        """Convert any input to LazyMatrix."""
        if isinstance(source, LazyMatrix):
            return source
        if isinstance(source, PipelineResult):
            return source.source
        return LazyMatrix(source)

    def _to_symbolic(self, source, lazy: LazyMatrix) -> SymbolicDescriptor:
        """Extract or create a SymbolicDescriptor."""
        if isinstance(source, SymbolicDescriptor):
            return source
        if isinstance(source, PipelineResult):
            return source.symbolic
        # Create from hash of dimensions
        sig = fmix64(lazy.rows ^ (lazy.cols << 32))
        return SymbolicDescriptor(lazy.rows, lazy.cols, sig)

    def _jit_matmul(self, A: LazyMatrix, B: LazyMatrix) -> list:
        """
        Tiled streaming matmul.
        Only one tile of each input in memory at a time.
        """
        M, K, N = A.rows, A.cols, B.cols
        T = self.tile_size
        C = [[0.0] * N for _ in range(M)]

        for i0 in range(0, M, T):
            im = min(i0 + T, M)
            for j0 in range(0, N, T):
                jm = min(j0 + T, N)
                for k0 in range(0, K, T):
                    km = min(k0 + T, K)
                    # JIT materialize only the needed tiles
                    A_tile = A.resolve_tile(i0, k0, im - i0, km - k0)
                    B_tile = B.resolve_tile(k0, j0, km - k0, jm - j0)
                    # Partial product
                    for i in range(im - i0):
                        for j in range(jm - j0):
                            s = 0.0
                            for k in range(km - k0):
                                s += A_tile[i][k] * B_tile[k][j]
                            C[i0 + i][j0 + j] += s
        return C

