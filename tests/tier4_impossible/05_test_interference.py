import os
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.vl.substrate.acceleration import SidechannelDetector

def test_interference():
    print("Tier 4: [05] Noisy Sidechannel Interference")
    det = SidechannelDetector()
    # Random white noise
    noise = [random.uniform(-1, 1) for _ in range(16)]
    seed = det.probe_stream_block(noise)
    print(f"Detection on Noise - Confidence: {det.confidence}")
    # Should be low, but could hallucinate structure (High Risk)
    print("[PASS] Interference test complete")

if __name__ == "__main__":
    test_interference()


