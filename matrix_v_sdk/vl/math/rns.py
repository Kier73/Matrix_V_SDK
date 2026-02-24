import math
from typing import List, Optional, Any

class VlAdaptiveRNS:
    """
    The Virtual Layer Adaptive RNS (Residue Number System).
    
    THEORY:
    RNS virtualizes large integer operations by decomposing a value into residues
    relative to a set of coprime bases (Prime Pool). 
    
    Properties:
    - Parallelism: Addition and Multiplication are bit-wise independent across primes.
    - No Carry: O(1) propagation for large-scale fixed-point operations.
    - Dynamic Range: Defined by the Product of Primes (M).
    
    Used by MMP_Engine to perform exact integer matmul before fixed-point scaling.
    """
    
    PRIME_POOL = [
        65447, 65449, 65479, 65497, 65519, 65521, 65437, 65423,
        65419, 65413, 65407, 65393, 65381, 65371, 65357, 65353,
        65327, 65323, 65309, 65293, 65287, 65269, 65267, 65257,
        65239, 65213, 65203, 65183, 65179, 65173, 65171, 65167,
    ]

    def __init__(self, count_or_primes: Any = 16):
        """Initializes a dynamic RNS basis."""
        if isinstance(count_or_primes, list):
            self.primes = count_or_primes
        else:
            count = count_or_primes if count_or_primes is not None else 16
            self.primes = self.PRIME_POOL[:min(count, len(self.PRIME_POOL))]
        self.M = 1
        for p in self.primes:
            self.M *= p
            
        self.precompute_crt()

    def precompute_crt(self):
        """Precomputes weights for the Chinese Remainder Theorem (CRT) reconstruction."""
        self.mi = []
        self.yi = []
        for p in self.primes:
            m_val = self.M // p
            self.mi.append(m_val)
            self.yi.append(self.mod_inverse(m_val, p))

    @staticmethod
    def mod_inverse(a: int, m: int) -> int:
        """Calculates modular multiplicative inverse via Extended Euclidean Algorithm."""
        def egcd(a, b):
            if a == 0:
                return (b, 0, 1)
            else:
                g, y, x = egcd(b % a, a)
                return (g, x - (b // a) * y, y)
        
        g, x, y = egcd(a, m)
        if g != 1:
            raise Exception('Modular inverse does not exist')
        else:
            return x % m

    def float_to_residues(self, data: Any, scale: int = 1000) -> List[List[int]]:
        """
        Projects float data into RNS residue space via fixed-point scaling.
        Used for bit-identical structural signatures.
        """
        results = []
        for val in data:
            # Handle both torch/numpy tensors and raw lists
            v_float = float(val.item()) if hasattr(val, 'item') else float(val)
            scaled = int(round(v_float * scale))
            results.append(self.decompose(scaled))
        return results

    def decompose(self, n: int) -> List[int]:
        """Project a large integer into the Residue Space."""
        return [n % p for p in self.primes]

    def reconstruct(self, residues: List[int]) -> int:
        """
        Re-materializes the value from its Residue projection using 
        Gauss's CRT formula: X = sum(ri * Mi * yi) mod M.
        """
        result = 0
        for r, m, y in zip(residues, self.mi, self.yi):
            result += r * m * y
        return result % self.M

# --- QUANTUM-ALIGNED RNS BASIS ---
# These primes are selected for their proximity to 2^16, 
# maximizing SIMD register utility in the Rust backend.
Q_RNS_PRIMES = [
    65447, 65449, 65479, 65497, 65519, 65521, 65437, 65423, 
    65419, 65413, 65407, 65393, 65381, 65371, 65357, 65353,
]

class ResidueValue:
    """
    A high-level wrapper for arithmetic in the Residue Space.
    Allows for O(1) carry-free operations on pseudo-tensors.
    """
    def __init__(self, residues: List[int]):
        self.residues = residues

    @classmethod
    def from_int(cls, value: int):
        return cls([value % p for p in Q_RNS_PRIMES])

    def __add__(self, other):
        return ResidueValue([(a + b) % p for a, b, p in zip(self.residues, other.residues, Q_RNS_PRIMES)])

    def __mul__(self, other):
        return ResidueValue([(a * b) % p for a, b, p in zip(self.residues, other.residues, Q_RNS_PRIMES)])

