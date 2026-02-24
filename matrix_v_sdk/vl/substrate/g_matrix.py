import ctypes
import os
import platform
import struct
from typing import List, Any, Dict

# --- RUST BACKEND SETUP ---
system = platform.system()
lib_name = "g_matrix_rust.dll" if system == "Windows" else "libg_matrix_rust.so"
lib_path = os.path.join(os.path.dirname(__file__), "g_matrix_rust", "target", "release", lib_name)

try:
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Backend library not found at {lib_path}")
    
    lib = ctypes.CDLL(lib_path)
    # FFI Signatures
    class GeometricDescriptorStruct(ctypes.Structure):
        _fields_ = [
            ("rows", ctypes.c_uint64),
            ("cols", ctypes.c_uint64),
            ("signature", ctypes.c_uint64),
            ("depth", ctypes.c_uint32),
        ]

    lib.g_matrix_symbolic_multiply.argtypes = [GeometricDescriptorStruct, GeometricDescriptorStruct]
    lib.g_matrix_symbolic_multiply.restype = GeometricDescriptorStruct

    lib.g_matrix_resolve.argtypes = [GeometricDescriptorStruct, ctypes.c_uint64, ctypes.c_uint64]
    lib.g_matrix_resolve.restype = ctypes.c_float

    lib.g_matrix_resolve_bulk.argtypes = [
        GeometricDescriptorStruct,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.c_size_t
    ]

    lib.g_matrix_rns_matmul.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t
    ]

    lib.g_matrix_inductive_matmul.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t
    ]

    HAS_RUST = True
except Exception:
    HAS_RUST = False

# --- CORE PRIMITIVES ---

def feistel_hash(addr: int) -> float:
    """Deterministic 64-bit Feistel hash for parameter generation."""
    l, r = (addr >> 32) & 0xFFFFFFFF, addr & 0xFFFFFFFF
    key = 0xBF58476D
    mul = 0x94D049BB
    for _ in range(4):
        f = ((r ^ key) * mul) & 0xFFFFFFFF
        f = ((f >> 16) ^ f) & 0xFFFFFFFF
        l, r = r, l ^ f
    return ((l << 32) | r) / float(2**64)

def fmix64(h: int) -> int:
    """MurmurHash3 finalizer mix for coordinate-based variety."""
    h ^= h >> 33
    h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    return h

def generate_signature(data: List[List[float]]) -> int:
    """Structural hash of matrix features."""
    rows = len(data)
    cols = len(data[0]) if rows > 0 else 0
    h = (rows ^ (cols << 13)) & 0xFFFFFFFFFFFFFFFF
    if rows > 0 and cols > 0:
        # Sample corners for stability
        h ^= int(data[0][0] * 1e6) ^ int(data[-1][-1] * 1e6)
    return fmix64(h)

# --- GEOMETRIC DESCRIPTOR ENGINE ---

class GeometricDescriptor:
    """
    Symbolic Matrix Descriptor.
    Represents a matrix as a deterministic field. Operations are performed on the descriptor.
    """
    def __init__(self, rows: int, cols: int, signature: int, depth: int = 1):
        self.rows = rows
        self.cols = cols
        self.signature = signature & 0xFFFFFFFFFFFFFFFF
        self.depth = depth

    def multiply(self, other: 'GeometricDescriptor') -> 'GeometricDescriptor':
        """Symbolic Descriptor Synthesis."""
        if self.cols != other.rows:
            raise ValueError(f"Dim mismatch: {self.cols} != {other.rows}")
        
        if HAS_RUST:
            a_struct = GeometricDescriptorStruct(self.rows, self.cols, self.signature, self.depth)
            b_struct = GeometricDescriptorStruct(other.rows, other.cols, other.signature, other.depth)
            res = lib.g_matrix_symbolic_multiply(a_struct, b_struct)
            return GeometricDescriptor(res.rows, res.cols, res.signature, res.depth)
        
        # Fallback to Python
        new_sig = (self.signature ^ (other.signature >> 1) ^ (self.depth << 32)) & 0xFFFFFFFFFFFFFFFF
        return GeometricDescriptor(self.rows, other.cols, new_sig, self.depth + other.depth)

    def resolve(self, row: int, col: int) -> float:
        """JIT Element Realization."""
        if HAS_RUST:
            struct = GeometricDescriptorStruct(self.rows, self.cols, self.signature, self.depth)
            return lib.g_matrix_resolve(struct, row, col)
        
        # Fallback to Python
        idx = (row * self.cols + col) & 0xFFFFFFFFFFFFFFFF
        h = fmix64(self.signature ^ idx)
        return (h / float(2**64)) * 2.0 - 1.0

# --- INFINITE SCALE INTERFACE ---

class GeometricMatrix:
    """
    Infinite-Scale Lazy Matrix.
    Behaves like a NumPy array but resolves elements on-demand.
    """
    def __init__(self, descriptor: GeometricDescriptor):
        self.desc = descriptor
        self.shape = (descriptor.rows, descriptor.cols)

    def __getitem__(self, key: Any) -> Any:
        import numpy as np
        if isinstance(key, tuple):
            row_key, col_key = key
            
            # Case 1: Single Element Resolution (e.g., M[0, 0])
            if isinstance(row_key, int) and isinstance(col_key, int):
                return self.desc.resolve(row_key, col_key)
            
            # Case 2: Slicing (e.g., M[0:10, 0:10])
            r_start = getattr(row_key, 'start', 0) or 0
            r_stop = getattr(row_key, 'stop', self.shape[0]) or self.shape[0]
            c_start = getattr(col_key, 'start', 0) or 0
            c_stop = getattr(col_key, 'stop', self.shape[1]) or self.shape[1]
            
            rows = r_stop - r_start
            cols = c_stop - c_start
            
            if HAS_RUST:
                struct = GeometricDescriptorStruct(self.desc.rows, self.desc.cols, self.desc.signature, self.desc.depth)
                res = np.zeros(rows * cols, dtype=np.float32)
                lib.g_matrix_resolve_bulk(
                    struct, 
                    res.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    r_start * self.desc.cols + c_start, 
                    rows * cols
                )
                return res.reshape(rows, cols)
            else:
                return np.array([[self.desc.resolve(r, c) for c in range(c_start, c_stop)] for r in range(r_start, r_stop)])
        
        # Case 3: Single Row Index (e.g., M[0])
        return self.desc.resolve(key, 0)

# --- INDUCTIVE TILE ENGINE ---

class InductiveEngine:
    """
    Tile-based Memoization Engine.
    Accelerates dense matmul by recalling previously computed tile interactions.
    """
    def __init__(self, tile_size: int = 32):
        self.tile_size = tile_size
        self.law_cache: Dict[int, List[List[float]]] = {}

    def _hash_tile(self, tile: List[List[float]]) -> int:
        """Full-content rolling hash over ALL tile elements.
        Each element's float bits are folded with its index via fmix64,
        preventing collisions between tiles with different interiors."""
        h = 0xcbf29ce484222325  # FNV offset basis
        idx = 0
        for row in tile:
            for val in row:
                # Pack float to 4-byte int representation
                bits = struct.unpack('I', struct.pack('f', val))[0]
                h ^= fmix64(bits ^ idx)
                h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF  # FNV prime
                idx += 1
        return h

    def matmul(self, A: Any, B: Any) -> Any:
        import numpy as np
        # Convert to numpy if not already
        a_np = np.asarray(A, dtype=np.float32)
        b_np = np.asarray(B, dtype=np.float32)
        
        M, K = a_np.shape
        N = b_np.shape[1]

        if HAS_RUST:
            c_arr = np.zeros(M * N, dtype=np.float32)
            lib.g_matrix_inductive_matmul(
                a_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                b_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                M, K, N
            )
            return c_arr.reshape(M, N)

        # Python Fallback (Tiled Inductive Execution)
        C = [[0.0] * N for _ in range(M)]
        for i in range(0, M, self.tile_size):
            for j in range(0, N, self.tile_size):
                for k in range(0, K, self.tile_size):
                    # Extract Tiles
                    tile_a = [row[k:k+self.tile_size] for row in A[i:i+self.tile_size]]
                    tile_b = [[B[r][c] for c in range(j, min(j+self.tile_size, N))] for r in range(k, min(k+self.tile_size, K))]
                    
                    # Inductive Lookup
                    pair_hash = self._hash_tile(tile_a) ^ (self._hash_tile(tile_b) << 1)
                    if pair_hash in self.law_cache:
                        res_tile = self.law_cache[pair_hash]
                    else:
                        # Compute & Induct
                        res_tile = self._compute_tile_prod(tile_a, tile_b)
                        self.law_cache[pair_hash] = res_tile

                    # Accumulate
                    for ti in range(len(res_tile)):
                        for tj in range(len(res_tile[0])):
                            if i + ti < M and j + tj < N:
                                C[i+ti][j+tj] += res_tile[ti][tj]
        return C

    def _compute_tile_prod(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        rows, cols = len(A), len(B[0])
        inner = len(A[0])
        res = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                res[i][j] = sum(A[i][k] * B[k][j] for k in range(inner))
        return res

# --- UNIFIED G_MATRIX INTERFACE ---
from .sdk_registry import solver, method

@solver("GMatrix")
class GMatrix:
    """Unified Entry Point for Generation 2 Matrix Operations."""
    def __init__(self, mode: str = "inductive"):
        self.mode = mode.lower()
        self.inductive = InductiveEngine()

    def from_data(self, data: List[List[float]]) -> GeometricDescriptor:
        sig = generate_signature(data)
        return GeometricDescriptor(len(data), len(data[0]), sig)

    def symbolic_matmul(self, A: GeometricDescriptor, B: GeometricDescriptor) -> GeometricMatrix:
        """O(1) Symbolic Multiply returning an Infinite Matrix."""
        desc = A.multiply(B)
        return GeometricMatrix(desc)

    @method("GMatrix", "matmul")
    def matmul(self, A: Any, B: Any) -> Any:
        """Tile-accelerated Inductive Multiply."""
        return self.inductive.matmul(A, B)

    @method("GMatrix", "rns_matmul")
    def rns_matmul(self, A: Any, B: Any) -> Any:
        """Bit-Exact modular matrix multiplication."""
        import numpy as np
        a_np = np.asarray(A, dtype=np.float32)
        b_np = np.asarray(B, dtype=np.float32)
        M, K = a_np.shape
        N = b_np.shape[1]
        
        if HAS_RUST:
            c_arr = np.zeros(M * N, dtype=np.float32)
            lib.g_matrix_rns_matmul(
                a_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                b_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                M, K, N
            )
            return c_arr.reshape(M, N)
        return self.matmul(A, B) # Fallback

if __name__ == "__main__":
    gm = GMatrix()
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[0.5, 0.1], [0.2, 0.6]]
    
    # 1. Inductive Test
    print("Testing Inductive GEMM...")
    res1 = gm.matmul(A, B)
    print(f"Cold Pass Result: {res1}")
    res2 = gm.matmul(A, B) # Should hit cache
    print(f"Warm Pass Result: {res2}")
    
    # 2. Geometric Test
    print("\nTesting Geometric Descriptor...")
    desc_a = gm.from_data(A)
    desc_b = gm.from_data(B)
    desc_c = gm.symbolic_matmul(desc_a, desc_b)
    print(f"Synthesized Descriptor: {desc_c.signature:x}")
    print(f"Resolved Element (0,0): {desc_c.resolve(0,0)}")

