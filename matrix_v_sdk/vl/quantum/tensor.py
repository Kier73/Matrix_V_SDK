import math
from typing import List

from ..math.rns import Q_RNS_PRIMES

class MPSNode:
    """
    Algebraic Matrix Product State (MPS) Node.
    O(n) storage for quantum states by leveraging RNS.
    """
    
    def __init__(self):
        self.residues = [0] * 16 # RNS physical index
        self.left_bond_mask = 0
        self.right_bond_mask = 0
        self.bond_dimension = 1

    def compress_bond(self, other: 'MPSNode'):
        """Analogous to Truncated SVD; identifies shared variety."""
        shared_mask = 0
        for i in range(16):
            if self.residues[i] == other.residues[i] and self.residues[i] != 0:
                shared_mask |= (1 << i)
        
        self.right_bond_mask |= shared_mask
        other.left_bond_mask |= shared_mask
        
        self.update_bond_dim()
        other.update_bond_dim()

    def update_bond_dim(self):
        # Actual Implementation: Chi = ceil(2^S)
        self.bond_dimension = math.ceil(2 ** self.entanglement_entropy())

    def entanglement_entropy(self) -> float:
        # Actual Implementation: Shannon Information Entropy of residues
        # S = -sum(p_i * log2(p_i))
        entropy = 0.0
        for r, p in zip(self.residues, Q_RNS_PRIMES):
            if r > 0:
                prob = r / p
                entropy -= prob * math.log2(prob)
        return entropy

