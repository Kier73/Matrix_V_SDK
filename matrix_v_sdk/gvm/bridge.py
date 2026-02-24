import ctypes
import os
import platform
from typing import Optional

# Configuration
_DEFAULT_DLL_PATH = os.path.join(os.path.dirname(__file__), "kernels", "gmem.dll")

class GvmBridge:
    """Low-level ctypes bridge to gmem.dll"""
    
    def __init__(self, dll_path: Optional[str] = None):
        if dll_path is None:
            dll_path = _DEFAULT_DLL_PATH
            
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"GVM Kernel not found at: {dll_path}")
            
        # Support DLL search paths on Windows
        if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(os.path.dirname(dll_path))
            
        self.lib = ctypes.CDLL(dll_path)
        self._setup_signatures()
        
    def _setup_signatures(self):
        # Core Lifecycle
        self.lib.gmem_create.argtypes = [ctypes.c_uint64]
        self.lib.gmem_create.restype = ctypes.c_void_p
        
        self.lib.gmem_destroy.argtypes = [ctypes.c_void_p]
        self.lib.gmem_destroy.restype = None
        
        # IO / Fetch
        self.lib.gmem_fetch_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.lib.gmem_fetch_f32.restype = ctypes.c_float
        
        self.lib.gmem_fetch_bulk_f32.argtypes = [
            ctypes.c_void_p, 
            ctypes.c_uint64, 
            ctypes.POINTER(ctypes.c_float), 
            ctypes.c_size_t
        ]
        self.lib.gmem_fetch_bulk_f32.restype = None
        
        self.lib.gmem_write_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_float]
        self.lib.gmem_write_f32.restype = None
        
        # Search
        self.lib.gmem_search_f32.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.lib.gmem_search_f32.restype = ctypes.c_uint64
        
        # Mirroring / Morphing
        self.lib.gmem_mirror_attach.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.gmem_mirror_attach.restype = None
        
        # Persistence
        self.lib.gmem_persistence_attach.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.gmem_persistence_attach.restype = ctypes.c_int
        
        # Trinity (If available in exported symbols - Checking headers)
        # Note: gmem_trinity.c symbols might be internal unless marked __declspec(dllexport)
        # We will assume they are available if part of the build.
        try:
            self.lib.g_inductive_resolve_sorted.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64
            ]
            self.lib.g_inductive_resolve_sorted.restype = ctypes.c_float
        except AttributeError:
             print("[BRIDGE] Warning: g_inductive_resolve_sorted not found in DLL.")

# Global instance for easy access
try:
    bridge = GvmBridge()
except Exception as e:
    print(f"[BRIDGE] Initialization error: {e}")
    bridge = None

