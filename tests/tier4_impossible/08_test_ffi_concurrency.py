import os
import sys
import threading
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import G_SeriesEngine

def test_ffi_concurrency():
    print("Tier 4: [08] Race Condition Stress on Rust FFI (Extreme)")
    engine = G_SeriesEngine()
    A = [[1.0]*4 for _ in range(4)]
    
    def stress():
        for _ in range(100):
            engine.multiply(A, A)
            
    threads = [threading.Thread(target=stress) for _ in range(32)] # High thread count
    for t in threads: t.start()
    for t in threads: t.join()
    print("[PASS] Rust FFI survived concurrency stress")

if __name__ == "__main__":
    test_ffi_concurrency()


