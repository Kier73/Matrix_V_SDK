import math
from ..math.primitives import vl_mask
from ..math.rns import Q_RNS_PRIMES

class VL_TopologicalState:
    """
    Simulates Topological Quantum Computing using Anyonic Braiding.
    Information is stored in the global topology of the RNS chain.
    """
    
    def __init__(self):
        self.crime_chain = list(Q_RNS_PRIMES)
        self.braid_signature = 0xCAFEBABE

    def braid(self, i: int, j: int):
        """Swaps positions of Anyons (Primes) to update topological Knot Invariant."""
        if i < len(self.crime_chain) and j < len(self.crime_chain):
            self.crime_chain[i], self.crime_chain[j] = self.crime_chain[j], self.crime_chain[i]
            self.recalculate_invariant()

    def recalculate_invariant(self):
        # Chained hashing (Non-Abelian order matters)
        sig = 0xCAFEBABE
        for prime in self.crime_chain:
            sig = vl_mask(sig, prime)
        self.braid_signature = sig

    def calculate_harmonic_invariant(self) -> float:
        real_sum = 0.0
        imag_sum = 0.0
        n = len(self.crime_chain)
        for k, p in enumerate(self.crime_chain):
            theta = 2.0 * math.pi * (p / 1000.0) * (k / n)
            real_sum += math.cos(theta)
            imag_sum += math.sin(theta)
        return math.sqrt(real_sum**2 + imag_sum**2)

