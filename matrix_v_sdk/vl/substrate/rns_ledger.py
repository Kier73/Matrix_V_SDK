"""
RNS Ledger — Memthematic I/O Gap 3B.

Exact RNS-based verification ledger for distributed exascale results.
Provides carry-free parallel verification without full reconstruction.

Ported from:
  - gmem_vrns.c (Torus Projection)
  - rns.rs (CRT Reconstruction)

Architecture:
  Dense Matrix C (m×n)
       ↓ Decompose
  RNS Ledger: [(addr, residues[8])] per element
       ↓ Verify
  CRT(residues) == quantized(C[r][c])   ← Exact parity check
"""

import struct
from typing import List, Tuple, NamedTuple



# Standard RNS moduli (shared with Rust RNSEngine)
MODULI = [251, 257, 263, 269, 271, 277, 281, 283]
DYNAMIC_RANGE = 1
for m in MODULI:
    DYNAMIC_RANGE *= m
# ≈ 2.7 × 10^19


class RNSFingerprint(NamedTuple):
    """An exact RNS fingerprint of a single value."""
    quantized: int       # Fixed-point integer representation
    residues: tuple      # 8 residues mod MODULI


class RNSLedger:
    """Exact verification ledger for a matrix result.

    Each element is stored as its RNS residues (8 × u16 = 16 bytes).
    Total footprint: 16 bytes/element (vs 8 bytes for raw f64).
    NOT for compression — for integrity verification of distributed results.
    """

    def __init__(self, rows: int, cols: int, scale: float = 1e6):
        self.rows = rows
        self.cols = cols
        self.scale = scale
        self._entries = {}  # (r, c) -> RNSFingerprint

    def record(self, r: int, c: int, value: float) -> RNSFingerprint:
        """Record an element's RNS fingerprint."""
        q = int(round(value * self.scale))
        # Handle negative values by shifting into positive range
        q_pos = q + (DYNAMIC_RANGE // 2)
        q_pos = q_pos % DYNAMIC_RANGE

        residues = tuple(q_pos % m for m in MODULI)
        fp = RNSFingerprint(quantized=q, residues=residues)
        self._entries[(r, c)] = fp
        return fp

    def verify(self, r: int, c: int, value: float) -> bool:
        """Verify that a value matches the recorded fingerprint.

        Uses CRT reconstruction for exact integer comparison.
        """
        fp = self._entries.get((r, c))
        if fp is None:
            return False

        q = int(round(value * self.scale))
        return q == fp.quantized

    def verify_residues(self, r: int, c: int, value: float) -> bool:
        """Verify via residue comparison (faster than CRT, still exact)."""
        fp = self._entries.get((r, c))
        if fp is None:
            return False

        q = int(round(value * self.scale))
        q_pos = q + (DYNAMIC_RANGE // 2)
        q_pos = q_pos % DYNAMIC_RANGE

        for i, m in enumerate(MODULI):
            if q_pos % m != fp.residues[i]:
                return False
        return True

    def torus_fingerprint(self, r: int, c: int) -> float:
        """Torus projection (from gmem_vrns.c).

        Returns a float in [0, 1) that summarizes the residue vector.
        Useful for quick approximate comparison.
        """
        fp = self._entries.get((r, c))
        if fp is None:
            return 0.0

        acc = 0.0
        for i, m in enumerate(MODULI):
            acc += fp.residues[i] / m
        return acc - int(acc)  # Fractional part (mod 1)

    @staticmethod
    def crt_reconstruct(residues: tuple) -> int:
        """Chinese Remainder Theorem reconstruction.

        Exact recovery of the original integer from its residues.
        Ported from rns.rs:from_residues().
        """
        result = 0
        for i in range(len(MODULI)):
            mi = MODULI[i]
            big_mi = DYNAMIC_RANGE // mi
            # Modular inverse via Extended Euclidean
            inv = _mod_inverse_small(big_mi % mi, mi)
            term = (residues[i] * big_mi * inv) % DYNAMIC_RANGE
            result = (result + term) % DYNAMIC_RANGE
        return result

    def memory_footprint_bytes(self) -> int:
        """Ledger memory footprint."""
        return len(self._entries) * 24  # 8 residues × 2 bytes + 8 bytes quantized

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "footprint_bytes": self.memory_footprint_bytes(),
            "dense_bytes": self.rows * self.cols * 8,
            "overhead_ratio": self.memory_footprint_bytes() / max(1, self.rows * self.cols * 8),
        }


def _mod_inverse_small(a: int, m: int) -> int:
    """Modular inverse for small moduli (< 300)."""
    # Brute force is fast for small primes
    a = a % m
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return 1


def record_matrix(matrix: List[List[float]],
                  scale: float = 1e6) -> RNSLedger:
    """Record an entire matrix into an RNS ledger."""
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    ledger = RNSLedger(rows, cols, scale)

    for r in range(rows):
        for c in range(cols):
            ledger.record(r, c, matrix[r][c])

    return ledger


def verify_matrix(matrix: List[List[float]],
                  ledger: RNSLedger) -> Tuple[bool, int, int]:
    """Verify an entire matrix against an RNS ledger.

    Returns (all_passed, pass_count, total).
    """
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    passes = 0
    total = rows * cols

    for r in range(rows):
        for c in range(cols):
            if ledger.verify_residues(r, c, matrix[r][c]):
                passes += 1

    return passes == total, passes, total


if __name__ == "__main__":
    print("=" * 60)
    print("RNS LEDGER — GAP 3B VERIFICATION")
    print("=" * 60)

    # Test 1: Simple matrix roundtrip
    matrix = [[1.5, -2.3, 0.0], [4.7, -1.1, 3.14]]
    ledger = record_matrix(matrix)
    ok, passes, total = verify_matrix(matrix, ledger)
    print(f"\n  Simple 2×3 Matrix:")
    print(f"    Verified: {passes}/{total} ({'PASS' if ok else 'FAIL'})")
    print(f"    Stats: {ledger.stats()}")

    # Test 2: CRT Reconstruction
    fp = ledger.record(0, 0, 1.5)
    recovered = RNSLedger.crt_reconstruct(fp.residues)
    q_expected = fp.quantized + (DYNAMIC_RANGE // 2)
    q_expected = q_expected % DYNAMIC_RANGE
    print(f"\n  CRT Roundtrip:")
    print(f"    Original quantized: {fp.quantized}")
    print(f"    Shifted:            {q_expected}")
    print(f"    CRT recovered:      {recovered}")
    print(f"    Match:              {recovered == q_expected}")

    # Test 3: Torus fingerprints
    torus_a = ledger.torus_fingerprint(0, 0)
    torus_b = ledger.torus_fingerprint(0, 1)
    print(f"\n  Torus Fingerprints:")
    print(f"    C[0,0]: {torus_a:.6f}")
    print(f"    C[0,1]: {torus_b:.6f}")

    # Test 4: Tamper detection
    tampered = [[1.5, -2.3, 0.0], [4.7, -1.1, 3.15]]  # 3.14 → 3.15
    ok, passes, total = verify_matrix(tampered, ledger)
    print(f"\n  Tamper Detection:")
    print(f"    Verified: {passes}/{total} ({'DETECTED' if not ok else 'MISSED'})")

    print("=" * 60)

