import math
import random

def low_rank_multiply(A, B, D=64):
    """Approximate Matmul via Random Projection (JL-Lemma)."""
    m, k = len(A), len(A[0])
    n = len(B[0])
    
    # 1. Generate Projection R (k x D)
    scale = math.sqrt(3.0 / D)
    R = [[0.0]*D for _ in range(k)]
    for i in range(k):
        for j in range(D):
            r = random.random()
            if r < 1/6: R[i][j] = scale
            elif r < 2/6: R[i][j] = -scale
            
    # 2. A_proj = A @ R (m x D)
    A_proj = [[sum(A[i][p] * R[p][j] for p in range(k)) for j in range(D)] for i in range(m)]
    
    # 3. B_proj = R^T @ B (D x n)
    B_proj = [[sum(R[p][i] * B[p][j] for p in range(k)) for j in range(n)] for i in range(D)]
    
    # 4. Result C = A_proj @ B_proj (m x n)
    # This remains O(m*D*n)
    C = [[sum(A_proj[i][p] * B_proj[p][j] for p in range(D)) for j in range(n)] for i in range(m)]
    return C

if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    B = [[0.5, 0.1], [0.2, 0.6]]
    C = low_rank_multiply(A, B, D=4)
    print(f"Approximate Product (D=4): {C}")

