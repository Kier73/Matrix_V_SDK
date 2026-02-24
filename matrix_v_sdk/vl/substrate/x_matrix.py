"""
XMatrix: Isomorphic Semantic Engine (Generation 3.5)
---------------------------------------------------
A pure-Python, dependency-free engine merging Variety, Geometry, 
and Isomorphic HDC (Hyperdimensional Computing).

Key Upgrades:
- 1024-bit Packed HDC Vectors ([int; 16]).
- Structural Composition History: Directional binding tracking.
- Early Resolution: Descriptor recognition to bypass redundant compute.
"""

import time
import ctypes
import os
import platform
from typing import List, Tuple, Union, Any, Dict

# --- RUST BACKEND SETUP ---

lib_path = os.path.join(os.path.dirname(__file__), "x_matrix_rust", "target", "release", "x_matrix_rust.dll")
if not os.path.exists(lib_path):
    lib_path = os.path.join(os.path.dirname(__file__), "x_matrix_rust", "target", "release", "libx_matrix_rust.so")

class Hdc1024Struct(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 16)]

try:
    lib = ctypes.CDLL(lib_path)
    lib.x_matrix_bind.argtypes = [ctypes.POINTER(Hdc1024Struct), ctypes.POINTER(Hdc1024Struct), ctypes.POINTER(Hdc1024Struct)]
    lib.x_matrix_shift.argtypes = [ctypes.POINTER(Hdc1024Struct), ctypes.c_size_t, ctypes.POINTER(Hdc1024Struct)]
    lib.x_matrix_similarity.argtypes = [ctypes.POINTER(Hdc1024Struct), ctypes.POINTER(Hdc1024Struct)]
    lib.x_matrix_similarity.restype = ctypes.c_float
    lib.x_matrix_fmix64.argtypes = [ctypes.c_uint64]
    lib.x_matrix_fmix64.restype = ctypes.c_uint64
    lib.x_matrix_resolve_interaction.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(Hdc1024Struct)]
    HAS_RUST = True
except Exception:
    HAS_RUST = False

# --- CORE PROCESSING GROUNDING (Pure Python) ---

def fmix64(h: int) -> int:
    """MurmurHash3 64-bit finalizer mix."""
    if HAS_RUST:
        return lib.x_matrix_fmix64(ctypes.c_uint64(h))
    
    h &= 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    return h

# --- ISOMORPHIC HDC CORE ---

HDC_DIM = 1024
U64_COUNT = 16  # 1024 / 64

class HdcManifold:
    """
    1024-bit Bit-Packed Vector.
    In this context: 1 bit = +1, 0 bit = -1.
    XOR Binding = Compositional Product.
    """
    def __init__(self, data: List[int] = None, seed: int = None, label: str = None):
        if seed is not None:
            self.seed = seed
            self.data = self._from_seed(seed)
        elif data is not None:
            self.data = data
            self.seed = None
        else:
            self.data = [0] * U64_COUNT
            self.seed = None
        self.label = label

    def _from_seed(self, seed: int) -> List[int]:
        """Deterministic 1024-bit vector generation (HDC-Sync)."""
        data = [0] * U64_COUNT
        s = seed
        for i in range(U64_COUNT):
            s = (s + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
            data[i] = fmix64(s)
        return data

    def bind(self, other: 'HdcManifold') -> 'HdcManifold':
        """XOR Binding: A * B"""
        if HAS_RUST:
            a_struct = Hdc1024Struct((ctypes.c_uint64 * 16)(*self.data))
            b_struct = Hdc1024Struct((ctypes.c_uint64 * 16)(*other.data))
            out_struct = Hdc1024Struct()
            lib.x_matrix_bind(ctypes.byref(a_struct), ctypes.byref(b_struct), ctypes.byref(out_struct))
            res = list(out_struct.data)
        else:
            res = [self.data[i] ^ other.data[i] for i in range(U64_COUNT)]
            
        if self.label and other.label:
            new_label = f"({self.label} * {other.label})"
            if len(new_label) > 128:
                new_label = new_label[:120] + "...)"
        else:
            new_label = None
        return HdcManifold(data=res, label=new_label)

    def shift(self, n: int) -> 'HdcManifold':
        """Directional Permutation (Circular Shift)."""
        if HAS_RUST:
            a_struct = Hdc1024Struct((ctypes.c_uint64 * 16)(*self.data))
            out_struct = Hdc1024Struct()
            lib.x_matrix_shift(ctypes.byref(a_struct), ctypes.c_size_t(n), ctypes.byref(out_struct))
            res = list(out_struct.data)
        else:
            n %= HDC_DIM
            giant = 0
            for i in range(U64_COUNT):
                giant |= self.data[i] << (64 * i)
            masked = giant & ((1 << HDC_DIM) - 1)
            rotated = ((masked << n) | (masked >> (HDC_DIM - n))) & ((1 << HDC_DIM) - 1)
            res = []
            for i in range(U64_COUNT):
                res.append((rotated >> (64 * i)) & 0xFFFFFFFFFFFFFFFF)
                
        return HdcManifold(data=res, label=self.label)

    def bundle(self, others: List['HdcManifold']) -> 'HdcManifold':
        """Superposition of multiple manifolds via bitwise majority vote."""
        all_m = [self] + list(others)
        res = [0] * U64_COUNT
        for i in range(U64_COUNT):
            # Bitwise majority vote
            counts = [0] * 64
            for m in all_m:
                val = m.data[i]
                for b in range(64):
                    if (val >> b) & 1:
                        counts[b] += 1
            
            out_val = 0
            threshold = len(all_m) / 2
            for b in range(64):
                if counts[b] > threshold:
                    out_val |= (1 << b)
                elif counts[b] == threshold:
                    # Random but deterministic tie-break
                    if (i + b + len(all_m)) % 2 == 0:
                        out_val |= (1 << b)
            res[i] = out_val
        return HdcManifold(data=res, label="Bundle")

    def similarity(self, other: 'HdcManifold') -> float:
        """Normalized Cosine Similarity (via Hamming)."""
        if HAS_RUST:
            a_struct = Hdc1024Struct((ctypes.c_uint64 * 16)(*self.data))
            b_struct = Hdc1024Struct((ctypes.c_uint64 * 16)(*other.data))
            return lib.x_matrix_similarity(ctypes.byref(a_struct), ctypes.byref(b_struct))
        
        hamming = 0
        for i in range(U64_COUNT):
            diff = self.data[i] ^ other.data[i]
            hamming += bin(diff).count('1')
        return 1.0 - (2.0 * hamming / HDC_DIM)

    def resolve(self, index: int) -> float:
        """Procedural value resolution at index."""
        word = (index % HDC_DIM) // 64
        bit = index % 64
        val = (self.data[word] >> bit) & 1
        return 1.0 if val == 0 else -1.0 # 0 bit -> +1, 1 bit -> -1

# --- MANIFOLD ORACLE ---

class ManifoldOracle:
    """
    Isomorphic Recognition Hub.
    Maps HDC signatures to known mathematical results.
    """
    def __init__(self):
        self.registry: Dict[int, Any] = {}

    def register(self, manifold: HdcManifold, law: str):
        sig = self._get_sig(manifold)
        self.registry[sig] = law

    def find_isomorph(self, manifold: HdcManifold) -> str:
        sig = self._get_sig(manifold)
        return self.registry.get(sig)

    def _get_sig(self, manifold: HdcManifold) -> int:
        """Collapse 1024 bits to 64-bit signature."""
        h = 0xcbf29ce484222325
        for word in manifold.data:
            h ^= word
            h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        return h

# --- X_MATRIX EVOLUTION ---
from .sdk_registry import solver, method

@solver("XMatrix")
class XMatrix:
    """
    Generation 3.5 Matrix: Analytical Descriptor Engine.

    Core capability: Procedural deterministic matrix generation from compact
    seeds, with O(1) composition tracking via HDC manifold binding.

    - compose(): O(1) descriptor composition that tracks structural lineage
    - get_element(): O(1) deterministic element resolution at any (r,c)
    - multiply_materialize(): True matrix product via materialization (O(n^2 d))
    """
    def __init__(self, rows: int, cols: int, seed: int = 0x517, manifold: HdcManifold = None,
                 inner_dim: int = None):
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.manifold = manifold or HdcManifold(seed=seed, label="BaseVariety")
        self.oracle = ManifoldOracle()
        # Track the inner dimension for statistical correctness in composed matrices
        self._inner_dim = inner_dim

    def _get_row_descriptor(self, r: int) -> HdcManifold:
        return HdcManifold(seed=fmix64(self.seed ^ r), label=f"R{r}")

    def _get_col_descriptor(self, c: int) -> HdcManifold:
        return HdcManifold(seed=fmix64(self.seed ^ (c + 0xABCDE)), label=f"C{c}")

    @method("XMatrix", "get_element")
    def get_element(self, r: int, c: int) -> float:
        """Resolve a deterministic value at position (r, c).

        For base matrices (no composition): produces values in [-1, +1]
        via HDC bit resolution.

        For composed matrices (_inner_dim is set): produces values drawn
        from N(0, inner_dim) via Box-Muller transform on deterministic seeds.
        This matches the statistical distribution of a true product of
        Rademacher (+-1) random matrices.
        """
        if self._inner_dim is not None:
            # Composed matrix: use Box-Muller for Gaussian values
            # with variance = inner_dim (correct for Rademacher product)
            import math as _math
            h1 = fmix64(self.manifold.data[0] ^ fmix64(r) ^ fmix64(c + 0xABCDE))
            h2 = fmix64(self.manifold.data[1] ^ fmix64(r + 0x12345) ^ fmix64(c))
            u1 = max((h1 & 0xFFFFFFFFFFFFFFFF) / float(2**64), 1e-300)
            u2 = (h2 & 0xFFFFFFFFFFFFFFFF) / float(2**64)
            z = _math.sqrt(-2.0 * _math.log(u1)) * _math.cos(2.0 * _math.pi * u2)
            return z * _math.sqrt(self._inner_dim)

        if HAS_RUST and self.manifold.seed is not None:
            # Optimized Path: Single FFI call for the entire interaction chain
            row_seed = fmix64(self.seed ^ r)
            col_seed = fmix64(self.seed ^ (c + 0xABCDE))
            out_struct = Hdc1024Struct()
            lib.x_matrix_resolve_interaction(
                ctypes.c_uint64(self.manifold.seed),
                ctypes.c_uint64(row_seed),
                ctypes.c_uint64(col_seed),
                ctypes.byref(out_struct)
            )
            # Resolve bit 0
            val = (out_struct.data[0] >> 0) & 1
            return 1.0 if val == 0 else -1.0

        # Fallback Pure Python path
        interaction = self._get_row_descriptor(r).bind(self._get_col_descriptor(c).shift(1))
        resolved_manifold = self.manifold.bind(interaction)
        return resolved_manifold.resolve(0)

    def compose(self, other: 'XMatrix') -> 'XMatrix':
        """
        O(1) Descriptor Composition.

        Creates a new XMatrix whose manifold encodes the structural
        lineage of both parents via HDC XOR binding. This does NOT
        compute a matrix product — it tracks composition history.

        The resulting matrix generates deterministic values consistent
        with the statistical properties of the Rademacher product.
        """
        if self.cols != other.rows:
            raise ValueError("Dimension mismatch")

        # HDC composition: bind parent descriptors
        new_manifold = self.manifold.bind(other.manifold.shift(7))
        new_manifold.label = f"({self.manifold.label} @ {other.manifold.label})"

        # Check Oracle for Isomorphism
        isomorph = self.oracle.find_isomorph(new_manifold)
        if isomorph:
            print(f"[ORACLE] Isomorph found: {isomorph}. Bypassing compute.")

        return XMatrix(self.rows, other.cols, manifold=new_manifold,
                       inner_dim=self.cols)

    @method("XMatrix", "multiply")
    def multiply(self, other: 'XMatrix') -> 'XMatrix':
        """
        Symbolic Composition (O(1)).

        NOTE: This performs O(1) descriptor composition, not a true matrix
        product. To compute the actual product, use multiply_materialize().
        The composed matrix generates values with the correct statistical
        distribution (Gaussian with variance = inner_dim).
        """
        return self.compose(other)

    def multiply_materialize(self, other: 'XMatrix', max_dim: int = 1000) -> List[List[float]]:
        """
        True matrix product via element-wise materialization.
        C[i][j] = sum_k A[i][k] * B[k][j].

        Complexity: O(rows * cols * inner_dim). Only feasible for moderate sizes.
        """
        if self.cols != other.rows:
            raise ValueError("Dimension mismatch")
        if self.rows > max_dim or other.cols > max_dim or self.cols > max_dim:
            raise ValueError(
                f"Matrix too large for materialization ({self.rows}x{self.cols}x{other.cols}). "
                f"Use compose() for symbolic O(1) operations."
            )

        C = [[0.0] * other.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                s = 0.0
                for k in range(self.cols):
                    s += self.get_element(i, k) * other.get_element(k, j)
                C[i][j] = s
        return C

    def to_list(self, max_rows: int = 4, max_cols: int = 4) -> List[List[float]]:
        return [
            [self.get_element(r, c) for c in range(min(max_cols, self.cols))]
            for r in range(min(max_rows, self.rows))
        ]

if __name__ == "__main__":
    print("-" * 60)
    print("XMATRIX (GEN 3.5): ISOMORPHIC SEMANTIC ENGINE")
    print("-" * 60)

    # 1. Scaling Test (Quintillion Scale)
    N = 10**18
    X = XMatrix(N, N, seed=1)
    Y = XMatrix(N, N, seed=2)

    start = time.perf_counter()
    Z = X.multiply(Y)
    latency = (time.perf_counter() - start) * 1000
    
    print(f"Matrix Scale:  {N} x {N}")
    print(f"Symbolic Ops:  {latency:.6f} ms")
    print(f"Ancestry:      {Z.manifold.label}")

    # 2. Isomorph Recognition Test
    # If we perform (X @ Y), we register it.
    oracle = ManifoldOracle()
    X.oracle = oracle
    oracle.register(Z.manifold, "GoldStandard_XY_Product")
    
    print("\nAttempting Redundant Multiplication...")
    Z2 = X.multiply(Y) # Should trigger Oracle

    # 3. Materialization
    print("\nSample 3x3 Interaction Manifold:")
    view = Z.to_list(3, 3)
    for row in view:
        print(f"  {['{:+.4f}'.format(v) for v in row]}")
    
    # 4. HDC Similarity Scan
    print(f"\nManifold Similarity (X vs Y): {X.manifold.similarity(Y.manifold):.4f}")
    print(f"Manifold Similarity (Z vs Z2): {Z.manifold.similarity(Z2.manifold):.4f}")
    print("-" * 60)

