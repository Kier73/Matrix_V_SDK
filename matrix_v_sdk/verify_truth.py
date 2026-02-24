import json
import math
import struct
from typing import List, Tuple

# MurmurHash3 fmix64
def fmix64(k: int) -> int:
    k &= 0xFFFFFFFFFFFFFFFF
    k ^= k >> 33
    k = (k * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    k ^= k >> 33
    k = (k * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    k ^= k >> 33
    return k

class VlAdaptiveRNS:
    PRIME_POOL = [
        65447, 65449, 65479, 65497, 65519, 65521, 65437, 65423,
        65419, 65413, 65407, 65393, 65381, 65371, 65357, 65353,
        65327, 65323, 65309, 65293, 65287, 65269, 65267, 65257,
        65239, 65213, 65203, 65183, 65179, 65173, 65171, 65167,
    ]
    def __init__(self, count: int = 32):
        self.primes = self.PRIME_POOL[:count]
    def synthesize(self, addr: int, seed: int) -> float:
        x = addr ^ seed
        h = seed
        for i, p in enumerate(self.primes):
            residue = x % p
            channel = (residue | (p << 16) | (i << 32)) & 0xFFFFFFFFFFFFFFFF
            h = fmix64(h ^ fmix64(channel))
        return h / 18446744073709551615.0

class FeistelProjector:
    def __init__(self, rounds: int = 4, key: int = 0xCBF29CE484222325):
        self.rounds = rounds
        self.key = key
    def project(self, l: int, r: int) -> Tuple[int, int]:
        for i in range(self.rounds):
            f = self.round_function(r, i ^ self.key)
            l, r = r, l ^ f
        return l, r
    def round_function(self, r: int, k: int) -> int:
        x = (r ^ k) & 0xFFFFFFFFFFFFFFFF
        x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x = (x >> 31) ^ x
        x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        x = (x >> 31) ^ x
        return x & 0xFFFFFFFFFFFFFFFF

class SymbolicDescriptor:
    def __init__(self, rows: int, cols: int, signature: int, depth: int = 1):
        self.rows = rows
        self.cols = cols
        self.signature = signature
        self.depth = depth
    def multiply(self, other: 'SymbolicDescriptor') -> 'SymbolicDescriptor':
        new_sig = (self.signature ^ (other.signature >> 1) ^ (self.depth << 32)) & 0xFFFFFFFFFFFFFFFF
        return SymbolicDescriptor(self.rows, other.cols, new_sig, self.depth + other.depth)
    def resolve(self, r: int, c: int) -> float:
        idx = (r * self.cols + c) & 0xFFFFFFFFFFFFFFFF
        h = fmix64(self.signature ^ idx)
        return h / 18446744073709551615.0 * 2.0 - 1.0

class HdcManifold:
    def __init__(self, seed: int):
        self.data = []
        s = seed
        for i in range(16):
            s = fmix64(s + i)
            self.data.append(s)

def generate_truth():
    truth = {
        "vRNS": [],
        "Feistel": [],
        "HDC": [],
        "Symbolic": []
    }
    
    # 1. vRNS Truth
    rns = VlAdaptiveRNS(32)
    for i in range(5):
        addr = 1000 + i
        seed = 0x1234
        val = rns.synthesize(addr, seed)
        truth["vRNS"].append({"addr": addr, "seed": seed, "expected": val})
        
    # 2. Feistel Truth
    feistel = FeistelProjector()
    for i in range(5):
        l, r = 0xAAAA + i, 0xBBBB + i
        nl, nr = feistel.project(l, r)
        truth["Feistel"].append({"l": l, "r": r, "expected_l": nl, "expected_r": nr})
        
    # 3. HDC Truth
    for i in range(3):
        seed = 0xDEADBEEF + i
        hdc = HdcManifold(seed)
        truth["HDC"].append({"seed": seed, "expected": hdc.data})
        
    # 4. Symbolic Truth
    s1 = SymbolicDescriptor(100, 100, 0x1234)
    s2 = SymbolicDescriptor(100, 100, 0x5678)
    s3 = s1.multiply(s2)
    truth["Symbolic"].append({
        "s1_sig": s1.signature,
        "s2_sig": s2.signature,
        "s3_expected_sig": s3.signature,
        "s3_expected_val_5_5": s3.resolve(5, 5)
    })
    
    with open("truth_data.json", "w") as f:
        json.dump(truth, f, indent=4)
    print("Truth data generated in truth_data.json")

    # 5. Benchmarks
    import time
    iterations_vrns = 1_000_000
    rns = VlAdaptiveRNS(32)
    start = time.time()
    _sum = 0.0
    for i in range(iterations_vrns):
        _sum += rns.synthesize(i, 0x1234)
    duration = time.time() - start
    print(f"Python vRNS: {iterations_vrns} iterations in {duration:.4f}s")
    
    iterations_sym = 100_000
    s = SymbolicDescriptor(1000, 1000, 0xABCD)
    start = time.time()
    _sum = 0.0
    for i in range(iterations_sym):
        _sum += s.resolve(i // 1000, i % 1000)
    duration = time.time() - start
    print(f"Python Symbolic Resolve: {iterations_sym} iterations in {duration:.4f}s")

if __name__ == "__main__":
    generate_truth()

