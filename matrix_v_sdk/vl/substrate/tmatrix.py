"""
T-Matrix Substrate
------------------
High-level abstractions for Morphological and Topological matrices.
Implements the weight-free foundations of TimesFM-X within the Matrix-V SDK.
"""
import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional

from .acceleration import TMatrixEngine

class TMatrix:
    """
    Holographic T-Matrix.
    Stores 'Morphological DNA' (Gielis parameters) instead of dense buffers.
    Scale: O(1) storage for O(N^2) manifolds.
    """
    def __init__(self, shape: Tuple[int, int], params: Optional[List[float]] = None):
        self.shape = shape
        # Default Gielis parameters: [m, a, b, n1, n2, n3]
        self.params = params or [4.0, 1.0, 1.0, 0.5, 0.5, 0.5]
        self.engine = TMatrixEngine()

    def materialize(self) -> torch.Tensor:
        """Project the T-Matrix into a dense PyTorch tensor using Holographic Mapping."""
        # Determine Hilbert order for the manifold
        order = int(math.ceil(math.log2(max(self.shape))))
        manifold = self.engine.project_holographic_manifold(self.params, self.shape, order)
        return torch.from_numpy(manifold).float()

    def get_rns_signature(self, count: int = 16) -> int:
        """
        Computes a bit-identical signature for the manifold.
        Guarantees cross-platform integrity via RNS residues.
        """
        from ..math.rns import VlAdaptiveRNS
        rns = VlAdaptiveRNS(count)
        manifold = self.materialize()
        residues = rns.float_to_residues(manifold.flatten())
        
        # Robust signature: Mix position and residue to avoid periodic cancellation
        sig = 0x517
        for i, res_set in enumerate(residues):
            for r in res_set:
                sig ^= (int(r) + i)
                sig = ((sig << 1) | (sig >> 63)) & 0xFFFFFFFFFFFFFFFF
        return sig

class T_MatrixVLinear(nn.Module):
    """
    Weight-Free Linear Layer (Drop-in Replacement).
    Generation: Weight(i, j) = Gielis(Hilbert(i, j, seed)).
    """
    def __init__(self, in_features: int, out_features: int, 
                 seed: int = 42, mode: str = 'ground'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed
        self.mode = mode # 'ground', 'dream', 'resonant'

        # Learnable DNA parameters (shifted by seed for diversity)
        dna_init = torch.tensor([4.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        if seed != 0:
            dna_init[0] += (seed % 100) / 10.0 # Shift 'm' parameter
        self.dna = nn.Parameter(dna_init)
        
        # Scaling to maintain variance (Xavier-style)
        self.register_buffer("scale", torch.tensor(1.0 / math.sqrt(in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.engine = TMatrixEngine()
        self._mask_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        m, k = self.out_features, self.in_features
        
        # 1. Project Ghost Weights from DNA
        # (Detached for pure-generative inference, or attached for DNA-learning)
        params = self.dna.detach().cpu().numpy().tolist()
        
        # Determine Hilbert order (used as seed for HDC mapping)
        order = int(math.ceil(math.log2(max(m, k))))
        
        # Holographic Projection: Uses HDC Mapping for Decorrelation
        r_grid = self.engine.project_holographic_manifold(params, (m, k), order)
        weights = torch.from_numpy(r_grid).to(device).float()
        
        # Standardization (Zero-mean, Unit-variance)
        # Ensures Xavier/Kaiming scaling behaves predictably
        weights = (weights - weights.mean()) / (weights.std() + 1e-9)
        
        # 2. Apply Resonant Shunting (Hilbert Wavefront)
        if self.mode == 'resonant':
            if self._mask_cache is None:
                # order is the smallest power of 2 encompassing the manifold
                order = int(math.ceil(math.log2(max(m, k))))
                full_mask = self.engine.get_hilbert_wavefront(order)
                # Crop to actual feature size
                self._mask_cache = torch.from_numpy(full_mask[:m, :k]).to(device).float()
            
            # Topological Filtering (90% shunting threshold)
            shunting_threshold = 0.9
            mask = (self._mask_cache > shunting_threshold).float()
            weights = weights * mask

        # 3. Standard Matrix Product (Accelerated via SDK backend if available)
        return torch.matmul(x, weights.t() * self.scale) + self.bias

