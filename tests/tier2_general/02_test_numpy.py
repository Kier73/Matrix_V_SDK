import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matrix_v_sdk.extensions.utils import to_list, from_list

def test_numpy():
    print("Tier 2: [02] NumPy Interop Stability")
    data = np.random.rand(10, 10).astype(np.float32)
    lst = to_list(data)
    assert isinstance(lst, list)
    assert len(lst) == 10
    
    back = from_list(lst, target_type='numpy')
    assert isinstance(back, np.ndarray)
    assert np.allclose(data, back)
    print("[PASS]")

if __name__ == "__main__":
    test_numpy()


