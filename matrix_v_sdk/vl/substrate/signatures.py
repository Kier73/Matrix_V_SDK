import math
import time
from typing import List, Optional, Tuple
from ..math.primitives import vl_signature

class SidechannelDetector:
    """
    Analyzes data streams to identify underlying Algebraic Laws.
    Bypasses I/O bottlenecks by 'Sign-Locking' onto generative manifolds.
    """
    
    def __init__(self, sensitivity: float = 0.95):
        self.sensitivity = sensitivity
        self.locked_signature: Optional[int] = None
        self.confidence: float = 0.0

    def probe_stream_block(self, block: List[float]) -> Optional[int]:
        """
        Analyzes a small block of data to detect varietal periodicity.
        Returns a Candidate Seed if a Law is detected.
        """
        if not block or len(block) < 4:
            return None
            
        n = len(block)
        sample = block[:min(n, 8)]
        
        # 1. Compute Varietal Signature
        raw_data = bytes([int(abs(x) * 255) % 256 for x in sample])
        sig = vl_signature(raw_data, seed=0x517)
        
        # 2. Statistical confidence: measure autocorrelation as proxy for structure
        mean = sum(sample) / len(sample)
        variance = sum((x - mean) ** 2 for x in sample) / len(sample)
        
        # Low variance = highly structured (constant/repeating)
        # High variance = random noise
        if variance < 1e-9:
            self.confidence = 1.0  # constant stream — fully locked
        else:
            # Normalized inverse variance, clamped to [0, 1]
            self.confidence = min(1.0, 1.0 / (1.0 + variance))
        
        if self.confidence >= self.sensitivity:
            self.locked_signature = sig
            return sig
            
        return None

    def predict_from_telemetry(self, latency_ms: float, throughput_mb: float) -> Optional[str]:
        """
        Adversarial sidechannel: Predicts compute task based on IO/Compute ratio patterns.
        """
        # MMP (Rectangular) has high throughput but steady latency.
        # P-Series (Lattice) has very low latency regardless of throughput.
        if latency_ms < 0.1:
            return "p_series"
        if throughput_mb > 500:
            return "mmp"
        return None

    def is_locked(self) -> bool:
        return self.confidence >= self.sensitivity

