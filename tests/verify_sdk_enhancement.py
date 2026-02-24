import sys
import os

# Add relevant paths
sys.path.append(os.path.abspath(os.curdir))
sys.path.append(os.path.abspath(os.path.join(os.curdir, 'Py')))

try:
    from matrix_v_sdk.vl_sdk.substrate.matrix import MatrixOmega
    import numpy as np

    def test_sdk_strategies():
        omega = MatrixOmega()
        
        # 1. Test Inductive (Tiled)
        tile = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        A_ind = tile + tile # 8x4 matrix with repetition
        B_ind = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        strategy_ind = omega.auto_select_strategy(A_ind, B_ind)
        print(f"Repetitive Data Strategy: {strategy_ind}")
        
        # 2. Test Spectral (Large Dense)
        A_dense = np.random.randn(128, 128).tolist()
        B_dense = np.random.randn(128, 128).tolist()
        strategy_spectral = omega.auto_select_strategy(A_dense, B_dense)
        print(f"Large Dense Strategy: {strategy_spectral}")
        
        # 3. Test MMP (Rectangular)
        A_rect = np.random.randn(10, 100).tolist()
        B_rect = np.random.randn(100, 10).tolist()
        strategy_mmp = omega.auto_select_strategy(A_rect, B_rect)
        print(f"Rectangular Strategy: {strategy_mmp}")

        # Functional Check (Spectral)
        print("\n[FUNCTIONAL CHECK] Spectral Matmul...")
        res = omega.compute_product(A_dense, B_dense)
        print(f"Spectral Result Shape: {len(res)}x{len(res[0])}")
        
        # Functional Check (Inductive)
        print("\n[FUNCTIONAL CHECK] Inductive Matmul...")
        res_ind = omega.compute_product(A_ind, B_ind)
        print(f"Inductive Result Shape: {len(res_ind)}x{len(res_ind[0])}")

    test_sdk_strategies()

except Exception as e:
    import traceback
    traceback.print_exc()


