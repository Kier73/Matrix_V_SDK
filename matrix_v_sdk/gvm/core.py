from typing import List, Union
import ctypes
from .bridge import bridge

class GenerativeMemory:
    """High-level interface for 1-Petabyte Virtual Data Substrate."""
    
    def __init__(self, seed: int = 0):
        if not bridge:
            raise RuntimeError("GVM Bridge not initialized. Check gmem.dll path.")
        self.seed = seed
        self.ctx = bridge.lib.gmem_create(seed)
        if not self.ctx:
            raise MemoryError("Failed to create GVM Context")
            
    def __del__(self):
        if hasattr(self, 'ctx') and self.ctx:
            bridge.lib.gmem_destroy(self.ctx)
            self.ctx = None
            
    def fetch(self, address: int) -> float:
        """Fetch a single float from the virtual space (O(1))."""
        return bridge.lib.gmem_fetch_f32(self.ctx, address)
        
    def write(self, address: int, value: float):
        """Write a value to the sparse RAM overlay."""
        bridge.lib.gmem_write_f32(self.ctx, address, value)
        
    def fetch_bulk(self, start_addr: int, count: int) -> List[float]:
        """Fetch a block of data using AVX-accelerated bulk kernels."""
        buffer = (ctypes.c_float * count)()
        bridge.lib.gmem_fetch_bulk_f32(self.ctx, start_addr, buffer, count)
        return list(buffer)

    def get_range(self, start_addr: int, count: int) -> List[float]:
        """Alias for fetch_bulk (Legacy compatibility)."""
        return self.fetch_bulk(start_addr, count)
        
    def search(self, target_value: float) -> int:
        """Constant-time (Interpolation) search for a value in the manifold."""
        return bridge.lib.gmem_search_f32(self.ctx, target_value)
        
    def attach_persistence(self, path: str):
        """Enable AOF logging and hydrate from existing state."""
        res = bridge.lib.gmem_persistence_attach(self.ctx, path.encode('utf-8'))
        if res != 0:
            raise IOError(f"Failed to attach persistence at {path}")
            
    def attach_mirror(self, other: 'GenerativeMemory'):
        """Mirror data from another GVM context."""
        bridge.lib.gmem_mirror_attach(self.ctx, other.ctx, 0) # Mode 0 = Mirror

    def resolve_manifold_2d(self, x: int, y: int, grid_size: int) -> float:
        """Resolve a 2D sorted manifold position (Trinity Synergy)."""
        try:
            return bridge.lib.g_inductive_resolve_sorted(self.ctx, x, y, grid_size)
        except AttributeError:
            # Fallback to pure-math implementation if not in DLL
            from .math import hilbert_xy_to_d
            d = hilbert_xy_to_d(grid_size, x, y)
            return self.fetch(d) # Simplified fallback

