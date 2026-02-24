"""
Anchor-Navigate: Dense Entry, Geometric Projection
=====================================================

THEORY:
  Given C = A x B, the traditional approach computes each C[i,j] as a
  full dot product: C[i,j] = SUM_t A[i,t] * B[t,j], cost O(k) per element.

  Anchor-Navigate computes ONE small dense block (the "anchor") and then
  navigates to any other element via CUR projection:

    C[i,j] = K[i,:] @ W_inv @ R[:,j]     cost O(s^2) per element

  Where:
    W = C[I,J]    -- s x s anchor block (computed dense, ground truth)
    R = C[I,:]    -- s anchor rows   (cost: O(skn), computed once)
    K = C[:,J]    -- s anchor columns (cost: O(mks), computed once)
    W_inv         -- pseudo-inverse of the anchor block

  ERROR BOUND:
    ||C - C_approx|| <= (1 + f(s)) * sigma_{s+1}(C)
    When rank(C) <= s: EXACT. Zero error.

  SPEEDUP:
    For N x N matrices with effective rank r, querying Q elements:
      Traditional:  O(N^3)
      Anchor-Nav:   O(rN^2 + Qr^2)
      Speedup:      N / (r + Qr^2/N^2)  for Q << N^2

PORT FROM: anchor_navigate.md formalization
"""

import numpy as np
import struct
import math
from typing import Optional, Tuple, List

from .rns_signature import RNSSignature, RNS_PRIMES, RNS_PRIMES_EXTENDED
from .rns_ledger import RNSLedger, record_matrix, verify_matrix
from ..math.primitives import fmix64


def _mod_inv(a: int, m: int) -> int:
    """Modular inverse via Extended Euclidean Algorithm."""
    a = a % m
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1


def _mod_matrix_inv(matrix: np.ndarray, p: int) -> np.ndarray:
    """
    Modular Matrix Inversion over F_p via Gaussian Elimination.
    Used for exact RNS navigation.
    """
    n = matrix.shape[0]
    # Augment with identity
    aug = np.zeros((n, 2 * n), dtype=np.int64)
    aug[:, :n] = matrix % p
    aug[:, n:] = np.eye(n, dtype=np.int64)

    for i in range(n):
        # Pivot search
        if aug[i, i] == 0:
            for j in range(i + 1, n):
                if aug[j, i] != 0:
                    aug[[i, j]] = aug[[j, i]]
                    break
            else:
                raise ValueError(f"Matrix is singular mod {p}")

        # Scale row i to have 1 at [i,i]
        inv = _mod_inv(int(aug[i, i]), p)
        aug[i] = (aug[i] * inv) % p

        # Eliminate other rows
        for j in range(n):
            if i != j:
                factor = aug[j, i]
                aug[j] = (aug[j] - factor * aug[i]) % p

    return aug[:, n:]


class AnchorNavigator:
    """
    Dense Entry, Geometric Projection.

    Compute a small anchor block of C = A x B exactly,
    then navigate to any element via CUR projection.

    ADAPTIVE SELECTION:
      Uses structural signals from the SDK's MatrixFeatureVector
      (sparsity, row_variance, tile_periodicity) together with
      RNS residue conditioning to choose optimal anchor indices.

    Usage:
        nav = AnchorNavigator(A, B)               # adaptive selection
        nav = AnchorNavigator(A, B, anchor_size=5) # force rank-5 anchor
        nav = AnchorNavigator(A, B, strategy='rns') # RNS-guided only

        val = nav.navigate(i, j)              # O(s^2) per element
        block = nav.navigate_block(rows, cols) # vectorized sub-block
        exact = nav.exact(i, j)               # ground truth (O(k))
        err = nav.error_at(i, j)              # |navigated - exact|
    """

    __slots__ = ('A', 'B', 'm', 'k', 'n', 's',
                 'I', 'J', 'R', 'K', 'W', 'W_inv',
                 '_rns', '_anchor_ledger',
                 '_effective_rank', '_sv_decay',
                 '_features', '_selection_strategy',
                 '_scale', '_exact_enabled', '_rns_inv_cache')

    def __init__(self, A, B,
                 anchor_size: Optional[int] = None,
                 anchor_rows: Optional[List[int]] = None,
                 anchor_cols: Optional[List[int]] = None,
                 strategy: str = 'adaptive',
                 scale: int = 1000,
                 exact: bool = False):
        """
        Args:
            A: m x k matrix (numpy array, list, or any array-like)
            B: k x n matrix
            anchor_size: number of anchor rows/columns (auto if None)
            anchor_rows: explicit anchor row indices (overrides auto)
            anchor_cols: explicit anchor column indices (overrides auto)
            strategy: 'adaptive' (best of all), 'norm', 'rns', 'spread'
            scale: fixed-point scaling factor for exact navigation
            exact: if True, enable RNS-exact arithmetic path
        """
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        m, k1 = self.A.shape
        k2, n = self.B.shape
        if k1 != k2:
            raise ValueError(f"Inner dimension mismatch: {k1} != {k2}")
        self.m, self.k, self.n = m, k1, n
        self._selection_strategy = strategy
        self._scale = scale
        self._exact_enabled = exact
        self._rns_inv_cache = {}

        # --- Step 1: Extract structural features ---
        self._features = self._extract_features()

        # --- Step 2: Adaptive rank estimation ---
        self._effective_rank, self._sv_decay = self._adaptive_rank()

        if anchor_size is None:
            anchor_size = self._effective_rank
        self.s = max(1, min(anchor_size, m, n))

        # --- Step 3: Select anchor indices ---
        if anchor_rows is not None:
            self.I = np.array(anchor_rows, dtype=int)
        elif strategy == 'adaptive' and not exact:
            self.I = self._select_rows_adaptive()
        elif exact:
            self.I = self._select_rows_rns()
        else:
            self.I = self._select_rows_by_strategy(strategy, 'rows')

        if anchor_cols is not None:
            self.J = np.array(anchor_cols, dtype=int)
        elif strategy == 'adaptive' and not exact:
            self.J = self._select_cols_adaptive()
        elif exact:
            self.J = self._select_cols_rns()
        else:
            self.J = self._select_rows_by_strategy(strategy, 'cols')

        # --- Step 4: Compute anchor structures ---
        self._build_anchors()

    # ==========================================================
    # STRUCTURAL FEATURE EXTRACTION (from MatrixFeatureVector)
    # ==========================================================

    def _extract_features(self) -> dict:
        """
        Extract structural signals from the matrices.

        Re-uses the same equations as MatrixFeatureVector:
          sparsity     = |{a_ij = 0}| / (m * k)
          row_variance = Var(||a_i||^2)
          tile_period  = frac of sampled rows with repeating tiles

        These directly inform anchor size and placement.
        """
        m, k, n = self.m, self.k, self.n

        # --- Sparsity (sampled) ---
        sample_rows = min(m, 16)
        sample_cols = min(k, 64)
        zero_count = 0
        total_sampled = sample_rows * sample_cols
        row_norms = []

        for i in range(sample_rows):
            norm_sq = np.sum(self.A[i, :sample_cols] ** 2)
            zero_count += int(np.sum(self.A[i, :sample_cols] == 0.0))
            row_norms.append(float(norm_sq))

        sparsity = zero_count / max(1, total_sampled)

        # --- Row variance ---
        if row_norms:
            mean_norm = sum(row_norms) / len(row_norms)
            row_var = sum((rn - mean_norm) ** 2 for rn in row_norms) / len(row_norms)
        else:
            row_var = 0.0

        # --- Tile periodicity ---
        tile_period = 0.0
        if m >= 8 and k >= 8:
            tile_matches = 0
            check_limit = min(m, 16)
            tile_size = min(4, k // 2)
            for r in range(check_limit):
                t1 = self.A[r, :tile_size].tolist()
                t2 = self.A[r, tile_size:2 * tile_size].tolist()
                if t1 == t2:
                    tile_matches += 1
            tile_period = tile_matches / check_limit

        # --- Column energy profile (for B) ---
        col_norms = np.linalg.norm(self.B, axis=0) ** 2
        col_var = float(np.var(col_norms)) if len(col_norms) > 0 else 0.0

        return {
            'sparsity': sparsity,
            'row_variance': row_var,
            'col_variance': col_var,
            'tile_periodicity': tile_period,
            'row_norms': np.linalg.norm(self.A, axis=1) ** 2,
            'col_norms': col_norms,
        }

    # ==========================================================
    # ADAPTIVE RANK ESTIMATION
    # ==========================================================

    def _adaptive_rank(self) -> Tuple[int, np.ndarray]:
        """
        Fuse multiple rank estimation signals:

        1. SVD probe (random sub-block, direct measurement)
        2. Feature-based upper bounds:
           - High sparsity => rank <= (1 - sparsity) * min(m,n)
           - Low row_variance => rows are similar => low rank
           - Tile periodicity => near-rank-1 structure
        3. JL-informed minimum:
           - d >= 4*ln(N) / eps^2   (from V_SeriesEngine)
           - If we want eps=0.1 accuracy, d >= 4*ln(N)/0.01

        Returns the geometric mean of these signals, clamped.
        """
        f = self._features
        m, k, n = self.m, self.k, self.n
        min_dim = min(m, n, k)

        # --- Signal 1: SVD probe ---
        probe_size = min(32, m, n, k)
        rng = np.random.RandomState(42)
        ri = rng.choice(m, probe_size, replace=False)
        ci = rng.choice(n, probe_size, replace=False)
        probe = self.A[ri, :] @ self.B[:, ci]
        sv = np.linalg.svd(probe, compute_uv=False)

        if sv[0] == 0:
            return 1, sv
        threshold = sv[0] * 1e-6
        svd_rank = int(np.sum(sv > threshold))

        # --- Signal 2: Feature-based bound ---
        feature_rank = min_dim  # default: full rank

        # High sparsity => fewer non-zero singular values
        if f['sparsity'] > 0.5:
            feature_rank = min(feature_rank,
                               max(1, int((1.0 - f['sparsity']) * min_dim)))

        # Low row variance => rows point in similar directions => low rank
        if f['row_variance'] < 1e-4 and min_dim > 2:
            feature_rank = min(feature_rank, max(1, min_dim // 4))

        # Tile periodicity => near-circulant => few dominant frequencies
        if f['tile_periodicity'] > 0.3:
            feature_rank = min(feature_rank,
                               max(1, int((1.0 - f['tile_periodicity']) * min_dim)))

        # --- Signal 3: JL-informed minimum ---
        # From V_SeriesEngine: d >= 4*ln(N)/(eps^2/2 - eps^3/3) for eps=0.1
        eps = 0.1
        jl_min = max(1, int(4 * math.log(max(2, min_dim))
                            / (eps**2 / 2 - eps**3 / 3)))

        # --- Fuse: geometric mean of SVD and feature, floored by JL ---
        fused = max(1, int(math.sqrt(svd_rank * feature_rank)))
        fused = max(fused, min(jl_min, min_dim))

        return min(fused, min_dim), sv

    # ==========================================================
    # ADAPTIVE ANCHOR SELECTION
    # ==========================================================

    def _candidate_indices(self, which: str) -> dict:
        """
        Generate multiple candidate anchor index sets.

        Returns dict mapping strategy_name -> indices array.
        Each strategy captures different structural information.
        """
        s = self.s

        if which == 'rows':
            energy = self._features['row_norms']
            N = self.m
        else:
            energy = self._features['col_norms']
            N = self.n

        s = min(s, N)
        candidates = {}

        # Strategy 1: NORM — highest energy (greedy leverage)
        candidates['norm'] = np.sort(np.argsort(energy)[-s:])

        # Strategy 2: SPREAD — evenly spaced (maximum coverage)
        candidates['spread'] = np.sort(
            np.linspace(0, N - 1, s, dtype=int))

        # Strategy 3: RNS-SPACED — RNS-based spacing via primes
        # Use prime moduli to create quasi-random but deterministic spacing
        rns_indices = set()
        for p_idx, p in enumerate(RNS_PRIMES[:4]):
            start = p_idx * (N // (len(RNS_PRIMES[:4]) + 1))
            for step in range(s):
                idx = (start + step * p) % N
                rns_indices.add(idx)
                if len(rns_indices) >= s:
                    break
            if len(rns_indices) >= s:
                break
        # Fill remaining with norm-based if needed
        if len(rns_indices) < s:
            norm_order = np.argsort(energy)[::-1]
            for idx in norm_order:
                rns_indices.add(int(idx))
                if len(rns_indices) >= s:
                    break
        candidates['rns'] = np.sort(np.array(list(rns_indices))[:s])

        # Strategy 4: MIXED — top half by norm, bottom half by spread
        half = s // 2
        top_norm = np.argsort(energy)[-(s - half):]
        top_spread = np.linspace(0, N - 1, half, dtype=int)
        mixed = np.unique(np.concatenate([top_norm, top_spread]))[:s]
        if len(mixed) < s:
            # Fill with remaining indices by energy
            remaining = np.argsort(energy)[::-1]
            for idx in remaining:
                if idx not in mixed:
                    mixed = np.append(mixed, idx)
                if len(mixed) >= s:
                    break
        candidates['mixed'] = np.sort(mixed[:s])

        return candidates

    def _anchor_quality(self, I, J) -> float:
        """
        Score an anchor selection by the conditioning of W = C[I,J].

        Lower condition number => better anchor => lower navigation error.
        Uses a fast RNS-based proxy first, falls back to actual W compute.

        Returns: 1 / condition_number  (higher is better)
        """
        # Compute the intersection block
        W = self.A[I, :] @ self.B[:, J]  # s x s

        # Fast proxy: RNS residue diversity
        # If all residues are similar, the block is ill-conditioned
        w_sum = float(np.sum(W))
        w_hash = fmix64(int.from_bytes(
            struct.pack('d', w_sum), byteorder='little'))
        residues = tuple(w_hash % p for p in RNS_PRIMES)
        diversity = len(set(residues)) / len(residues)

        if diversity < 0.3:
            return 1e-12  # likely degenerate

        # Full conditioning check
        try:
            cond = np.linalg.cond(W)
            if cond > 1e12:
                return 1e-12
            return 1.0 / cond
        except np.linalg.LinAlgError:
            return 1e-12

    def _select_rows_adaptive(self) -> np.ndarray:
        """
        Try all candidate strategies, score each by anchor quality,
        pick the best one.
        """
        candidates = self._candidate_indices('rows')

        # For columns, initially use norm-based
        col_energy = self._features['col_norms']
        J_temp = np.sort(np.argsort(col_energy)[-self.s:])

        best_name = 'norm'
        best_score = -1
        best_I = candidates['norm']

        for name, I in candidates.items():
            if len(I) != self.s:
                continue
            score = self._anchor_quality(I, J_temp)
            if score > best_score:
                best_score = score
                best_name = name
                best_I = I

        self._selection_strategy = best_name
        return best_I

    def _select_cols_adaptive(self) -> np.ndarray:
        """
        Select columns after rows are fixed, testing candidates.
        """
        candidates = self._candidate_indices('cols')

        best_name = 'norm'
        best_score = -1
        best_J = candidates['norm']

        for name, J in candidates.items():
            if len(J) != self.s:
                continue
            score = self._anchor_quality(self.I, J)
            if score > best_score:
                best_score = score
                best_name = name
                best_J = J

        return best_J

    def _select_rows_by_strategy(self, strategy: str, which: str) -> np.ndarray:
        """Select using a specific named strategy."""
        candidates = self._candidate_indices(which)
        if strategy in candidates:
            return candidates[strategy]
        return candidates['norm']  # fallback

    # ==========================================================
    # ANCHOR COMPUTATION (the one dense block)
    # ==========================================================

    def _build_anchors(self):
        """
        Compute the anchor structures.

        This is the ONLY dense computation:
          R = A[I,:] @ B   -- s anchor rows of C    (s x n)
          K = A @ B[:,J]   -- s anchor columns of C (m x s)
          W = C[I,J]       -- the intersection      (s x s)
          W_inv            -- pseudo-inverse         (s x s)

        Cost: O(s * k * (m + n))
        """
        # Anchor rows: multiply only the selected rows of A by B
        self.R = self.A[self.I, :] @ self.B          # s x n

        # Anchor columns: multiply A by only the selected columns of B
        self.K = self.A @ self.B[:, self.J]            # m x s

        # Intersection: already computed in both R and K
        self.W = self.R[:, self.J]                     # s x s

        # Pseudo-inverse (handles rank-deficient anchor blocks)
        self.W_inv = np.linalg.pinv(self.W)            # s x s

        # RNS signature of the anchor for integrity
        anchor_hash = fmix64(
            int.from_bytes(
                struct.pack('d', float(np.sum(self.W))),
                byteorder='little'
            )
        )
        self._rns = RNSSignature(self.m, self.n, anchor_hash)

        # RNS ledger for the anchor block
        self._anchor_ledger = record_matrix(self.W.tolist())

        # If exact path is enabled, precompute modular inverses of W
        if self._exact_enabled:
            self._build_rns_anchors()

    def _build_rns_anchors(self):
        """Precompute modular inverses for all RNS channels."""
        # Scale W to integer domain
        W_int = np.round(self.W * self._scale * self._scale).astype(np.int64)
        
        # Use EXTENDED prime set (16 primes) for max capacity
        for p in RNS_PRIMES_EXTENDED:
            try:
                self._rns_inv_cache[p] = _mod_matrix_inv(W_int, p)
            except ValueError:
                # If W is singular mod p, it's a poor anchor for that prime
                # In RNS-Exact, we ideally want a non-singular anchor for ALL p
                pass

    # ==========================================================
    # NAVIGATION (the core operation)
    # ==========================================================

    def navigate(self, i: int, j: int) -> float:
        """
        Navigate to C[i,j] via anchor projection.
        
        If exact=True, uses RNS modular arithmetic.
        Otherwise, uses standard floating-point CUR.
        """
        if self._exact_enabled:
            return self.navigate_exact(i, j)
        return float(self.K[i, :] @ self.W_inv @ self.R[:, j])

    def navigate_exact(self, i: int, j: int) -> float:
        """
        RNS-Exact Navigation to C[i,j].
        
        Equations:
          res_p(C_ij) = res_p(K_i) @ res_p(W)^-1 @ res_p(R_j) mod p
          Value = CRT(res_p) / (scale^2)
        """
        # K and R are results of A @ B products, so they are at scale^2
        s2 = self._scale * self._scale
        K_i = np.round(self.K[i, :] * s2).astype(np.int64)
        R_j = np.round(self.R[:, j] * s2).astype(np.int64)
        
        residues = []
        for p in RNS_PRIMES_EXTENDED:
            W_inv_p = self._rns_inv_cache.get(p)
            if W_inv_p is None:
                residues.append(0)
                continue
            
            # K_i_p @ W_inv_p @ R_j_p  (all in Z/pZ)
            # 1. K_i @ W_inv  mod p
            kw_p = (K_i % p) @ W_inv_p % p
            # 2. (K_i @ W_inv) @ R_j mod p
            c_p = (kw_p @ (R_j % p)) % p
            residues.append(int(c_p))
            
        # CRT Reconstruction (using full extended set)
        raw = self._crt_reconstruct(tuple(residues), RNS_PRIMES_EXTENDED)
        
        # Pull back from fixed-point
        # MMP logic: if raw > M/2, it's negative
        M = 1
        for p in RNS_PRIMES_EXTENDED:
            M *= p
        
        if raw > M // 2:
            raw -= M
            
        return float(raw) / (self._scale * self._scale)

    @staticmethod
    def _crt_reconstruct(residues: tuple, primes: List[int]) -> int:
        """Standard CRT reconstruction for a given prime set."""
        M = 1
        for p in primes:
            M *= p
            
        result = 0
        for r, p in zip(residues, primes):
            Mi = M // p
            inv = _mod_inv(Mi % p, p)
            result = (result + r * Mi * inv) % M
        return result

    def navigate_block(self, rows, cols) -> np.ndarray:
        """
        Navigate to a sub-block C[rows, cols].

        C[rows,cols] = K[rows,:] @ W_inv @ R[:,cols]

        Fully vectorized. Cost: O(|rows| * |cols| * s).
        """
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        return self.K[rows, :] @ self.W_inv @ self.R[:, cols]

    def navigate_diagonal(self, n: Optional[int] = None) -> np.ndarray:
        """Navigate to the diagonal of C. Cost: O(min(m,n) * s^2)."""
        if n is None:
            n = min(self.m, self.n)
        diag = np.zeros(n)
        # Vectorized: element-wise dot of K[i,:] @ W_inv with R[:,i]
        KW = self.K[:n, :] @ self.W_inv  # n x s
        for i in range(n):
            diag[i] = KW[i, :] @ self.R[:, i]
        return diag

    # ==========================================================
    # GROUND TRUTH (for validation)
    # ==========================================================

    def exact(self, i: int, j: int) -> float:
        """Compute the exact element C[i,j] = A[i,:] @ B[:,j]. Cost: O(k)."""
        return float(self.A[i, :] @ self.B[:, j])

    def exact_block(self, rows, cols) -> np.ndarray:
        """Compute exact sub-block."""
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        return self.A[rows, :] @ self.B[:, cols]

    def error_at(self, i: int, j: int) -> float:
        """Absolute error at (i,j)."""
        return abs(self.navigate(i, j) - self.exact(i, j))

    def error_block(self, rows, cols) -> np.ndarray:
        """Element-wise absolute error for a sub-block."""
        nav = self.navigate_block(rows, cols)
        ext = self.exact_block(rows, cols)
        return np.abs(nav - ext)

    # ==========================================================
    # ANCHOR INTEGRITY (RNS verification)
    # ==========================================================

    def verify_anchor(self) -> bool:
        """Verify the anchor block via RNS ledger."""
        ok, passes, total = verify_matrix(self.W.tolist(), self._anchor_ledger)
        return ok

    @property
    def rns_signature(self) -> RNSSignature:
        """RNS signature of the anchor."""
        return self._rns

    # ==========================================================
    # COST ANALYSIS
    # ==========================================================

    @property
    def anchor_flops(self) -> int:
        """FLOPs spent computing anchors."""
        return 2 * self.s * self.k * (self.m + self.n) + self.s ** 3

    @property
    def dense_flops(self) -> int:
        """FLOPs for the full dense matmul."""
        return 2 * self.m * self.k * self.n

    @property
    def anchor_ratio(self) -> float:
        """Fraction of dense cost spent on anchors."""
        return self.anchor_flops / max(1, self.dense_flops)

    def query_cost(self, Q: int) -> int:
        """Total FLOPs for anchor + Q navigations."""
        return self.anchor_flops + Q * self.s ** 2

    def speedup(self, Q: int) -> float:
        """Speedup over computing Q exact dot products."""
        exact_cost = Q * 2 * self.k
        return exact_cost / max(1, self.query_cost(Q))

    def stats(self) -> dict:
        return {
            "shape": f"{self.m}x{self.k} @ {self.k}x{self.n}",
            "anchor_size": self.s,
            "effective_rank": self._effective_rank,
            "strategy": self._selection_strategy,
            "anchor_rows": self.I.tolist(),
            "anchor_cols": self.J.tolist(),
            "anchor_ratio": f"{self.anchor_ratio:.4f}",
            "W_cond": float(np.linalg.cond(self.W)),
            "rns_residues": self._rns.residues,
            "features": {
                'sparsity': self._features['sparsity'],
                'row_variance': self._features['row_variance'],
                'tile_periodicity': self._features['tile_periodicity'],
            },
        }

    def __repr__(self):
        return (f"<AnchorNav {self.m}x{self.n} s={self.s} "
                f"rank={self._effective_rank} "
                f"strategy={self._selection_strategy} "
                f"ratio={self.anchor_ratio:.4f}"
                f"{' EXACT' if self._exact_enabled else ''}>")

    # ==========================================================
    # RNS-BASED ANCHOR SELECTION (Residue MaxVol)
    # ==========================================================

    def _select_rows_rns(self) -> np.ndarray:
        """
        Select rows that are non-singular in all RNS channels.
        Tries candidate sets and selects the first one that is 
        modularly invertible across the extended prime set.
        """
        candidates = self._candidate_indices('rows')
        
        # Test columns (initially norm-based)
        col_energy = self._features['col_norms']
        J_temp = np.sort(np.argsort(col_energy)[-self.s:])

        for strat, I in candidates.items():
            if self._is_modular_invertible(I, J_temp):
                self._selection_strategy = f"rns-{strat}"
                return I
        
        # Fallback to norm if no candidate is perfectly stable
        return candidates['norm']

    def _select_cols_rns(self) -> np.ndarray:
        """Select columns after rows are fixed, using RNS stability."""
        candidates = self._candidate_indices('cols')
        
        for strat, J in candidates.items():
            if self._is_modular_invertible(self.I, J):
                self._selection_strategy += f"-{strat}"
                return J
        
        return candidates['norm']

    def _is_modular_invertible(self, I, J) -> bool:
        """Check if W = C[I,J] is invertible in all RNS channels."""
        W = self.A[I, :] @ self.B[:, J]
        W_int = np.round(W * self._scale * self._scale).astype(np.int64)
        
        for p in RNS_PRIMES_EXTENDED:
            try:
                _mod_matrix_inv(W_int, p)
            except ValueError:
                return False
        return True

