"""
VL Substrate: Exascale matrix engines and Memthematic I/O.
"""

# --- Core Engines ---
from .g_matrix import GMatrix, GeometricMatrix, GeometricDescriptor
from .x_matrix import XMatrix, HdcManifold
from .anchor import AnchorNavigator

# --- RNS Signature (ring-homomorphic composition) ---
from .rns_signature import RNSSignature, RNS_PRIMES, RNS_PRIMES_EXTENDED

# --- Memthematic I/O ---
from .tile_collapser import TileLaw, collapse, resolve, verify_collapse_parity
from .manifold_fitter import ManifoldDescriptor, fit_manifold, verify_manifold_parity
from .rns_ledger import RNSLedger, record_matrix, verify_matrix

# --- Bounded Descriptors (error exclusion) ---
from .bounded import BoundedDescriptor, ProcessShape, LogicShape, ErrorExclusion
from .bounded import bounded_from_seed, bounded_matmul

# --- Pipeline ---
from .pipeline import MemthematicPipeline, PipelineResult, LazyMatrix

