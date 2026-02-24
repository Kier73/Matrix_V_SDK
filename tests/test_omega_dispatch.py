import sys
import os
sys.path.append(os.path.abspath(os.curdir))
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega
import numpy as np

def test_omega_dispatch_logic():
    print("\n--- [TEST] MatrixOmega Adaptive Dispatch Logic ---")
    omega = MatrixOmega()
    
    # 1. Small Dense Matrix (Should use "dense")
    A_small = np.random.randn(16, 16).tolist()
    B_small = np.random.randn(16, 16).tolist()
    strategy_small = omega.auto_select_strategy(A_small, B_small)
    print(f"Small (16x16) -> Strategy: {strategy_small} (Expected: dense)")
    
    # 2. Large Sparse Matrix (Should use "spectral" or "qmatrix")
    N = 600
    A_sparse = np.zeros((N, N))
    # Add some sparse elements
    for i in range(0, N, 10):
        A_sparse[i, i] = 1.0
    A_sparse = A_sparse.tolist()
    B_dense = np.random.randn(N, N).tolist()
    strategy_sparse = omega.auto_select_strategy(A_sparse, B_dense)
    print(f"Large Sparse ({N}x{N}) -> Strategy: {strategy_sparse}")
    
    # 3. High Tile Periodicity (Should use "inductive")
    P = 128
    tile = np.random.randn(8, 8)
    A_periodic = np.tile(tile, (P//8, P//8)).tolist()
    B_periodic = np.tile(tile, (P//8, P//8)).tolist()
    strategy_periodic = omega.auto_select_strategy(A_periodic, B_periodic)
    print(f"Periodic ({P}x{P}) -> Strategy: {strategy_periodic} (Expected: inductive)")
    
    # 4. Rectangular Bottleneck (Should use "mmp")
    A_rect = np.random.randn(10, 500).tolist()
    B_rect = np.random.randn(500, 10).tolist()
    strategy_rect = omega.auto_select_strategy(A_rect, B_rect)
    print(f"Rectangular Bottleneck (10x500x10) -> Strategy: {strategy_rect} (Expected: mmp)")

if __name__ == "__main__":
    test_omega_dispatch_logic()


