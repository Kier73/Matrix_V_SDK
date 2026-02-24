import numpy as np
import sys
import os
from .acceleration import V_SeriesEngine

class KinematicEngine:
    """
    [DEPRECATED] Kinematic Matmul Bridge.
    
    Now reroutes to V_SeriesEngine (Spectral Projector) for improved 
    stability and performance. Original Ana-Kata projection is 
    maintained as a symbolic ghost strategy.
    """
    def __init__(self, seed=42):
        self._fallback = V_SeriesEngine(epsilon=0.1)

    def multiply(self, A: np.ndarray, B: np.ndarray, threshold: float = 0.8, check_accuracy: bool = True) -> np.ndarray:
        """Reroutes to V_SeriesEngine as per SDK depreciation policy."""
        # Convert to list for V_Series if necessary, or just use numpy directly if V_Series supports it
        # V_SeriesEngine expects List[List[float]] based on previous view
        A_list = A.tolist() if isinstance(A, np.ndarray) else A
        B_list = B.tolist() if isinstance(B, np.ndarray) else B
        
        C_list = self._fallback.multiply(A_list, B_list)
        return np.array(C_list)

if __name__ == "__main__":
    # Internal test
    engine = KinematicEngine()
    a = np.random.randn(64, 64).astype(np.float32)
    b = np.random.randn(64, 64).astype(np.float32)
    c = engine.multiply(a, b)
    print(f"Kinematic Matmul (Fallback) Output Shape: {c.shape}")

