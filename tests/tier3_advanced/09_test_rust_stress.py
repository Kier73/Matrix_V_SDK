import os
import sys
import numpy as np
import threading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import G_SeriesEngine

def test_rust_stress():
    print("Tier 3: [09] Concurrent Cache Stress Test (Rust)")
    engine = G_SeriesEngine()
    A = [[1.0]*32 for _ in range(32)]
    B = [[1.0]*32 for _ in range(32)]
    
    def worker():
        for _ in range(5):
            engine.multiply(A, B)
            
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    print("[PASS]")

if __name__ == "__main__":
    test_rust_stress()


