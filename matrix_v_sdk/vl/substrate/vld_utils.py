import hashlib
import struct
import math
from typing import List, Tuple, Union, Any

class DeterministicHasher:
    """ Maps arbitrary noisy data into a fixed 256-bit coordinate space. """
    @staticmethod
    def hash_data(data: Any) -> int:
        if isinstance(data, str):
            encoded = data.encode()
        elif isinstance(data, list):
            try:
                encoded = struct.pack(f'{len(data)}d', *data)
            except:
                encoded = str(data).encode()
        elif isinstance(data, bytes):
            encoded = data
        else:
            encoded = str(data).encode()
        return int(hashlib.sha256(encoded).hexdigest(), 16)

class FeistelMemoizer:
    """ Symmetric Feistel Cipher for stable 64-bit Law seeds. """
    def __init__(self, rounds: int = 4):
        self.rounds = rounds
        self.key = 0xBF58476D

    def project_to_seed(self, coordinate_256: int) -> int:
        folded = (coordinate_256 >> 128) ^ coordinate_256
        addr = folded & ((1 << 128) - 1)
        l, r = (addr >> 64) & 0xFFFFFFFFFFFFFFFF, addr & 0xFFFFFFFFFFFFFFFF
        for _ in range(self.rounds):
            f = ((r ^ self.key) * 0xCBF29CE484222325) & 0xFFFFFFFFFFFFFFFF
            f = ((f >> 32) ^ f) & 0xFFFFFFFFFFFFFFFF
            l, r = r, l ^ f
        return (l << 64) | r

class ArchetypeEngine:
    """ Procedural Asset Resolver for JIT realization. """
    def __init__(self, seed: int = 0xABC):
        self.seed = seed
        self.feistel = FeistelMemoizer()

    def resolve_entry(self, path: str) -> dict:
        coord = int(hashlib.md5(path.encode()).hexdigest(), 16)
        v_offset = 0xF000000000000000 | (self.feistel.project_to_seed(coord) & 0xFFFFFFFFFF)
        is_dir = path.endswith('/')
        size = 0 if is_dir else (coord % (1024**4)) 
        return {"name": path.split('/')[-1] or path.split('/')[-2], "offset": v_offset, "size": size, "is_dir": is_dir}

