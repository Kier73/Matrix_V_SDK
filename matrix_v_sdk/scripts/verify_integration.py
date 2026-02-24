import sys
import os
import numpy as np

# Add the SDK root to path
sdk_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.append(sdk_root)

# Correct path for substrate imports
substrate_path = os.path.abspath(os.path.join(os.getcwd(), "vl", "substrate"))
sys.path.append(substrate_path)

try:
    from matrix_v_sdk.vl.substrate.matrix import MatrixOmega, XMatrix, PrimeMatrix
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    # Try alternate import for standalone testing
    try:
        from matrix import MatrixOmega, XMatrix, PrimeMatrix
        print("Fallback imports successful.")
    except ImportError as e2:
        print(f"Fallback import also failed: {e2}")
        sys.exit(1)

def test_integration():
    omega = MatrixOmega()
    
    # 1. Test Standard Matrix (should use spectral or adaptive block)
    print("\nTest 1: Standard Numerical Matrix")
    a = np.random.rand(128, 128).tolist()
    b = np.random.rand(128, 128).tolist()
    res1 = omega.compute_product(a, b)
    print(f"  Result Shape: {len(res1)}x{len(res1[0])}")

    # 2. Test XMatrix (Symbolic Descriptor)
    print("\nTest 2: XMatrix Symbolic Composition")
    x1 = XMatrix(100, 100, seed=1)
    x2 = XMatrix(100, 100, seed=2)
    res2 = omega.compute_product(x1, x2)
    print(f"  Result Type: {type(res2)}")
    print(f"  Result Matrix rows/cols: {res2.rows}x{res2.cols}")

    # 3. Test PrimeMatrix (Analytical Divisor)
    print("\nTest 3: PrimeMatrix Symbolic Composition")
    p1 = PrimeMatrix(50, 50)
    p2 = PrimeMatrix(50, 50)
    res3 = omega.compute_product(p1, p2)
    print(f"  Result Type: {type(res3)}")
    print(f"  Result Matrix depth: {getattr(res3, 'depth', 'N/A')}")

    print("\nVerification Complete.")

if __name__ == "__main__":
    test_integration()

