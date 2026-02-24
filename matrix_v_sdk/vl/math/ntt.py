from typing import Tuple, List
from .primitives import C_MAGIC

# Goldilocks Prime P = 2^64 - 2^32 + 1
P_GOLDILOCKS = 0xFFFFFFFF00000001

class NTTMorphism:
    """
    Bit-Exact NTT Morphism.
    Supports O(1) law resolution and Universal Law synthesis.
    """
    
    @staticmethod
    def multiply_mod(a: int, b: int) -> int:
        return (a * b) % P_GOLDILOCKS

    @staticmethod
    def add_mod(a: int, b: int) -> int:
        return (a + b) % P_GOLDILOCKS

    @staticmethod
    def synthesize_product_law(seed_a: int, seed_b: int) -> int:
        """Pointwise law synthesis in the NTT domain."""
        return NTTMorphism.multiply_mod(seed_a, seed_b)

    @staticmethod
    def resolve_value_at(seed: int, coord: Tuple[int, int]) -> int:
        """Resolve value from the deterministic variety field at (x, y)."""
        x, y = coord
        term1 = NTTMorphism.multiply_mod(seed, x)
        return NTTMorphism.add_mod(term1, y)

    @staticmethod
    def resolve_from_law_256(law_seed: List[int], input_variety: List[int], output_index: int, seed: int) -> int:
        """
        Resolve a bit-exact output from a 256-bit law seed.
        The core "Instantaneous Materialization" operation.
        """
        # Mix 256-bit (4x64) seeds down to 64-bit for NTT domain
        ls_mix = law_seed[0] ^ law_seed[1] ^ law_seed[2] ^ law_seed[3]
        iv_mix = input_variety[0] ^ input_variety[1] ^ input_variety[2] ^ input_variety[3]
        
        # Combined address = law ⊗ input ⊗ index
        combined = NTTMorphism.synthesize_product_law(ls_mix, iv_mix)
        addr = ((combined << 16) | (output_index & 0xFFFF)) & 0xFFFFFFFFFFFFFFFF
        
        # Deterministic hash for final value
        return (addr * C_MAGIC + seed) & 0xFFFFFFFFFFFFFFFF

