import os
import sys
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.signatures import SidechannelDetector

def test_sidechannel():
    print("Tier 3: [05] Fuzzy Locking Confidence Audit")
    det = SidechannelDetector()
    # High resonance signal (periodic sine wave)
    signal = [math.sin(i * 0.1) for i in range(16)]
    seed = det.probe_stream_block(signal)
    print(f"Seed: {seed}, Confidence: {det.confidence}")
    # The detector should compute a non-zero confidence for structured data
    assert det.confidence > 0.0, "Expected non-zero confidence"
    print("[PASS]")

if __name__ == "__main__":
    test_sidechannel()


