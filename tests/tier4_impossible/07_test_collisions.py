import os
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.math.primitives import fmix64

def test_collisions():
    print("Tier 4: [07] Hash Collision Probability Loop")
    # Birthday problem check on fmix64 (truncated)
    hashes = set()
    collisions = 0
    for i in range(100000):
        h = fmix64(i)
        if h in hashes:
            collisions += 1
        hashes.add(h)
    
    print(f"Collisions in 100k hashes: {collisions}")
    assert collisions == 0 # Murmur finalizer should be near perfect for this range
    print("[PASS]")

if __name__ == "__main__":
    test_collisions()


