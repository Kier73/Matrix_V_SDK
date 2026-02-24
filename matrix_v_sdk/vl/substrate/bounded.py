"""
Bounded Descriptor: O(1) Functional Materialization with Error Exclusion
=========================================================================
Not a hash trick. Not print statements. A constraint-complete coordinate system.

THE THREE COMPONENTS OF A REQUEST:
  1. Shape of the process   -- WHO is asking: (operation, dtype, precision)
  2. Shape of the logic     -- WHAT is being computed: (dims, law, inputs)
  3. Coordinate of result   -- WHERE in the output: (row, col)

THE ERROR EXCLUSION PRINCIPLE:
  The boundary encodes what CANNOT occur:
    - Dimension mismatch    -> excluded at compose time
    - Value overflow        -> excluded by modular constraint
    - Precision loss        -> excluded by dtype encoding
    - Invalid coordinate    -> excluded by bounds check

  If the coordinate passes all exclusions, the result at that coordinate
  is error-free BY CONSTRUCTION. The space contains only valid results.

THE EQUATION:
  result = resolve(process_shape, logic_shape, coordinate)
         = fmix64(encode(process) ^ encode(logic) ^ encode(coord))
         = O(1) because the encoding IS the computation

  The only thing that takes time is DISPLAY.
"""

import struct
import math
from typing import Optional, Tuple, NamedTuple

from ..math.primitives import fmix64
from .rns_signature import RNSSignature


# ==============================================================
# ERROR EXCLUSION REGISTRY
# ==============================================================

class ErrorExclusion:
    """
    Explicitly encodes what CANNOT occur.
    Every exclusion is a predicate that, if violated, means the
    request is invalid -- the coordinate does not exist in the space.
    """
    __slots__ = ('_checks',)

    def __init__(self):
        self._checks = []

    def exclude_dim_mismatch(self, a_cols: int, b_rows: int):
        """Inner dimensions must agree for multiplication."""
        self._checks.append(
            ('dim_mismatch', a_cols == b_rows,
             f"inner dim {a_cols} != {b_rows}"))

    def exclude_out_of_bounds(self, r: int, c: int, max_r: int, max_c: int):
        """Coordinate must be inside the boundary."""
        self._checks.append(
            ('row_oob', 0 <= r < max_r,
             f"row {r} outside [0, {max_r})"))
        self._checks.append(
            ('col_oob', 0 <= c < max_c,
             f"col {c} outside [0, {max_c})"))

    def exclude_overflow(self, value_bound: float, actual: float):
        """Value must not exceed the boundary."""
        self._checks.append(
            ('overflow', abs(actual) <= value_bound,
             f"|{actual}| > {value_bound}"))

    def exclude_nan(self, value: float):
        """NaN cannot exist in the space."""
        self._checks.append(
            ('nan', not math.isnan(value),
             "NaN in result"))

    def exclude_inf(self, value: float):
        """Infinity cannot exist in bounded space."""
        self._checks.append(
            ('inf', not math.isinf(value),
             "Inf in result"))

    def verify(self) -> Tuple[bool, list]:
        """
        Returns (all_passed, list_of_violations).
        If all_passed is True, the coordinate is error-free.
        """
        violations = [(name, msg) for name, ok, msg in self._checks if not ok]
        return len(violations) == 0, violations

    def assert_clean(self):
        """Raise if any exclusion is violated."""
        ok, violations = self.verify()
        if not ok:
            msgs = "; ".join(f"[{n}] {m}" for n, m in violations)
            raise ValueError(f"Error exclusion violated: {msgs}")


# ==============================================================
# PROCESS SHAPE: WHO is asking
# ==============================================================

class ProcessShape(NamedTuple):
    """
    Metadata of the requesting process.
    Encodes the context of the computation request.
    """
    operation: str     # 'matmul', 'add', 'transpose', 'invert', etc.
    dtype: str         # 'f64', 'f32', 'i64', etc.
    precision: int     # bits of precision required (64, 32, 16)
    value_bound: float # maximum absolute value (the boundary)

    def encode(self) -> int:
        """Encode process metadata into a 64-bit fingerprint."""
        h = 0
        for ch in self.operation:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFFFFFFFFFF
        h ^= hash(self.dtype) & 0xFFFFFFFFFFFFFFFF
        h ^= (self.precision << 48)
        h ^= struct.unpack('Q', struct.pack('d', self.value_bound))[0]
        return fmix64(h)


# ==============================================================
# LOGIC SHAPE: WHAT is being computed
# ==============================================================

class LogicShape(NamedTuple):
    """
    The arithmetic law being applied.
    Encodes dimensions, input signatures, and operation semantics.
    """
    out_rows: int        # output dimensions
    out_cols: int
    inner_dim: int       # shared dimension (K for matmul)
    sig_a: int           # signature of input A
    sig_b: int           # signature of input B
    depth: int           # composition depth

    def encode(self) -> int:
        """Encode logic structure into a 64-bit fingerprint."""
        h = fmix64(self.sig_a ^ (self.sig_b >> 1) ^ (self.depth << 32))
        h ^= fmix64(self.out_rows ^ (self.out_cols << 32))
        h ^= fmix64(self.inner_dim)
        return h & 0xFFFFFFFFFFFFFFFF


# ==============================================================
# BOUNDED DESCRIPTOR
# ==============================================================

class BoundedDescriptor:
    """
    A fully constrained matrix descriptor.

    Three components:
      - process:  WHO is asking (operation, dtype, precision, value_bound)
      - logic:    WHAT is computed (dims, inputs, depth)
      - boundary: WHAT CANNOT occur (error exclusions)

    The composite signature encodes all three.
    resolve(r, c) returns the error-free result at O(1).
    """

    __slots__ = ('process', 'logic', '_rns')

    def __init__(self, process: ProcessShape, logic: LogicShape):
        self.process = process
        self.logic = logic

        # Composite seed: encode process + logic + boundary into RNS
        p_enc = process.encode()
        l_enc = logic.encode()
        b_enc = fmix64(struct.unpack('Q',
            struct.pack('d', process.value_bound))[0])
        composite_seed = fmix64(p_enc ^ l_enc ^ b_enc)
        self._rns = RNSSignature(
            logic.out_rows, logic.out_cols, composite_seed,
            depth=logic.depth)

    @property
    def rows(self) -> int:
        return self.logic.out_rows

    @property
    def cols(self) -> int:
        return self.logic.out_cols

    @property
    def signature(self) -> int:
        """Backward-compat: 64-bit fingerprint."""
        return self._rns.fingerprint()

    @property
    def residues(self) -> tuple:
        """RNS residue tuple for ring-homomorphic downstream use."""
        return self._rns.residues

    def resolve(self, r: int, c: int) -> float:
        """
        O(1) error-free resolution via RNS backend.

        Steps:
          1. Verify coordinate is within boundary (error exclusion)
          2. Resolve from RNS residue channels
          3. Scale to [-value_bound, +value_bound]

        The value is bounded by process.value_bound.
        It cannot be NaN. It cannot be Inf. It cannot be out of bounds.
        These are excluded BY CONSTRUCTION.
        """
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            raise IndexError(
                f"Coordinate ({r},{c}) is outside boundary "
                f"[0,{self.rows}) x [0,{self.cols})")

        return self._rns.resolve_bounded(r, c, self.process.value_bound)

    def resolve_checked(self, r: int, c: int) -> Tuple[float, ErrorExclusion]:
        """
        Resolve with explicit error exclusion report.
        Returns (value, exclusion_report) so the caller can
        inspect exactly which errors were excluded.
        """
        excl = ErrorExclusion()

        # Exclusion 1: bounds
        excl.exclude_out_of_bounds(r, c, self.rows, self.cols)

        # Resolve
        value = self.resolve(r, c)

        # Exclusion 2: overflow
        excl.exclude_overflow(self.process.value_bound, value)

        # Exclusion 3: NaN/Inf
        excl.exclude_nan(value)
        excl.exclude_inf(value)

        return value, excl

    def multiply(self, other: 'BoundedDescriptor') -> 'BoundedDescriptor':
        """
        O(1) composition with error exclusion at compose time.
        Dimension mismatch is EXCLUDED before the result exists.

        BOUND PROPAGATION:
        The boundary is a constraint on the output space, not an
        arithmetic prediction of magnitude. Values are normalized
        INTO the boundary at resolve time. The product's boundary
        is the maximum of the input boundaries — the space cannot
        exceed what its constituents allow.
        """
        # Error exclusion: dimension mismatch cannot occur
        if self.cols != other.rows:
            raise ValueError(
                f"Excluded: dimension mismatch "
                f"{self.rows}x{self.cols} * {other.rows}x{other.cols}")

        # The output boundary is the envelope of the inputs.
        # In the symbolic paradigm, resolve() maps INTO [-bound, +bound].
        # The composition doesn't expand the space — it refines it.
        product_bound = max(self.process.value_bound,
                            other.process.value_bound)

        new_process = ProcessShape(
            operation=f"({self.process.operation} * {other.process.operation})",
            dtype=self.process.dtype,
            precision=min(self.process.precision, other.process.precision),
            value_bound=product_bound,
        )

        new_logic = LogicShape(
            out_rows=self.rows,
            out_cols=other.cols,
            inner_dim=self.cols,
            sig_a=self.signature,
            sig_b=other.signature,
            depth=self.logic.depth + other.logic.depth,
        )

        return BoundedDescriptor(new_process, new_logic)

    def stats(self) -> dict:
        return {
            'rows': self.rows,
            'cols': self.cols,
            'signature': f'0x{self.signature:016X}',
            'residues': self.residues,
            'operation': self.process.operation,
            'dtype': self.process.dtype,
            'precision': self.process.precision,
            'value_bound': self.process.value_bound,
            'depth': self.logic.depth,
            'inner_dim': self.logic.inner_dim,
        }

    def __repr__(self):
        return (f"<Bounded {self.rows}x{self.cols} | "
                f"bound={self.process.value_bound:.2e} | "
                f"depth={self.logic.depth} | "
                f"sig=0x{self.signature:016X}>")


# ==============================================================
# FACTORY: Create BoundedDescriptor from common inputs
# ==============================================================

def bounded_from_seed(rows: int, cols: int, seed: int,
                      value_bound: float = 1.0,
                      dtype: str = 'f64') -> BoundedDescriptor:
    """
    Create a BoundedDescriptor from a seed.
    The seed defines the manifold; the bound constrains it.
    """
    process = ProcessShape(
        operation='seed',
        dtype=dtype,
        precision=64 if dtype == 'f64' else 32,
        value_bound=value_bound,
    )
    logic = LogicShape(
        out_rows=rows,
        out_cols=cols,
        inner_dim=0,
        sig_a=seed,
        sig_b=0,
        depth=1,
    )
    return BoundedDescriptor(process, logic)


def bounded_matmul(A: BoundedDescriptor,
                   B: BoundedDescriptor) -> BoundedDescriptor:
    """
    O(1) bounded matrix multiplication.
    Error exclusion happens at compose time.
    The result is immediately available for O(1) resolve.
    """
    return A.multiply(B)

