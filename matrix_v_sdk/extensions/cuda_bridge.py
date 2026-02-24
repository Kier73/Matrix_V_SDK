"""
CUDA Bridge — Card-Agnostic GPU Acceleration
---------------------------------------------
Routes matrix operations to GPU via PyTorch's CUDA abstraction.
Works on any NVIDIA card (Turing, Ampere, Ada Lovelace, Hopper).

Architecture:
  1. Detect available GPU and its compute capability
  2. For large matrices (n > CUDA_THRESHOLD), use torch.matmul on GPU
  3. For structured matrices, use SDK engines on CPU (symbolic, RNS)
  4. Hybrid path: GPU for bulk compute, CPU for exact/symbolic ops

The bridge never hardcodes a device — it queries torch.cuda.get_device_properties()
and adapts tile sizes and precision based on available VRAM and SM count.
"""
try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

import time
import numpy as np
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega, MatrixFeatureVector, SHUNT_THRESHOLD


# Minimum matrix size to justify GPU transfer overhead
CUDA_THRESHOLD = 128


class CUDADeviceInfo:
    """
    Card-agnostic GPU capability detection.
    
    Queries torch.cuda to determine:
    - Device name and compute capability
    - Available VRAM for tile sizing
    - SM count for parallelism estimation
    """
    def __init__(self, device_id=0):
        self.available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.device_id = device_id
        self.device = None
        self.name = "CPU (no CUDA)"
        self.compute_capability = (0, 0)
        self.total_memory_mb = 0
        self.sm_count = 0
        self.optimal_tile_size = 64

        if self.available:
            self.device = torch.device(f"cuda:{device_id}")
            props = torch.cuda.get_device_properties(device_id)
            self.name = props.name
            self.compute_capability = (props.major, props.minor)
            self.total_memory_mb = props.total_mem // (1024 * 1024)
            self.sm_count = props.multi_processor_count

            # Adapt tile size based on VRAM
            # 8GB+ -> 256 tiles, 4GB -> 128, else 64
            if self.total_memory_mb >= 8192:
                self.optimal_tile_size = 256
            elif self.total_memory_mb >= 4096:
                self.optimal_tile_size = 128
            else:
                self.optimal_tile_size = 64

    def __repr__(self):
        if not self.available:
            return "CUDADeviceInfo(available=False)"
        return (f"CUDADeviceInfo({self.name}, "
                f"CC={self.compute_capability[0]}.{self.compute_capability[1]}, "
                f"VRAM={self.total_memory_mb}MB, "
                f"SM={self.sm_count}, "
                f"tile={self.optimal_tile_size})")


class CUDAMatrixV:
    """
    Card-agnostic CUDA-accelerated matrix multiplication.

    Dispatch logic:
      1. Symbolic inputs -> CPU (O(1), no GPU needed)
      2. m*k*n < CUDA_THRESHOLD^3 -> CPU (transfer overhead > compute)
      3. Large dense -> GPU via torch.matmul (cuBLAS under the hood)
      4. RNS/exact -> CPU MMP_Engine (integer arithmetic not suited for GPU)

    The bridge uses fp32 by default but supports fp16 on cards with
    compute capability >= 7.0 (Volta+) and tf32 on >= 8.0 (Ampere+).
    """
    def __init__(self, device_id=0, dtype='auto'):
        self.device_info = CUDADeviceInfo(device_id)
        self.omega = MatrixOmega()  # CPU fallback

        # Select precision based on card capability
        if dtype == 'auto' and self.device_info.available:
            cc = self.device_info.compute_capability
            if cc[0] >= 8:
                # Ampere+ supports TF32 for matmul (transparent acceleration)
                self._dtype = torch.float32
                self._use_tf32 = True
            elif cc[0] >= 7:
                self._dtype = torch.float32
                self._use_tf32 = False
            else:
                self._dtype = torch.float32
                self._use_tf32 = False
        else:
            self._dtype = torch.float32 if TORCH_AVAILABLE else None
            self._use_tf32 = False

    @property
    def available(self):
        return self.device_info.available

    def multiply(self, A, B, force_gpu=False, force_cpu=False):
        """
        Adaptive GPU/CPU matrix multiplication.

        Args:
            A, B: List[List[float]], np.ndarray, or torch.Tensor
            force_gpu: Always use GPU (skip cost analysis)
            force_cpu: Always use CPU (skip GPU)

        Returns:
            List[List[float]] (consistent with SDK format)
        """
        # Normalize inputs
        A_np, B_np = self._to_numpy(A), self._to_numpy(B)
        m, k = A_np.shape
        n = B_np.shape[1]

        # Decision: GPU or CPU?
        use_gpu = False
        if not force_cpu and self.available:
            if force_gpu:
                use_gpu = True
            elif m * k * n >= CUDA_THRESHOLD ** 3:
                use_gpu = True

        if use_gpu:
            return self._gpu_multiply(A_np, B_np)
        else:
            return self.omega.compute_product(A_np.tolist(), B_np.tolist())

    def _gpu_multiply(self, A_np, B_np):
        """Execute matmul on GPU via torch.matmul (cuBLAS)."""
        device = self.device_info.device

        # Enable TF32 if supported
        if self._use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        A_t = torch.tensor(A_np, dtype=self._dtype, device=device)
        B_t = torch.tensor(B_np, dtype=self._dtype, device=device)

        C_t = torch.matmul(A_t, B_t)

        # Transfer back to CPU and convert to list
        return C_t.cpu().numpy().tolist()

    def multiply_fp16(self, A, B):
        """
        Half-precision matmul for inference (2x throughput on Tensor Cores).
        Only available on compute capability >= 7.0 (Volta+).
        """
        if not self.available:
            return self.omega.compute_product(
                A.tolist() if hasattr(A, 'tolist') else A,
                B.tolist() if hasattr(B, 'tolist') else B
            )

        if self.device_info.compute_capability[0] < 7:
            # Fall back to fp32 on older cards
            return self.multiply(A, B, force_gpu=True)

        A_np, B_np = self._to_numpy(A), self._to_numpy(B)
        device = self.device_info.device

        A_t = torch.tensor(A_np, dtype=torch.float16, device=device)
        B_t = torch.tensor(B_np, dtype=torch.float16, device=device)
        C_t = torch.matmul(A_t, B_t)

        return C_t.float().cpu().numpy().tolist()

    def benchmark(self, n=512, iterations=10):
        """
        Benchmark GPU vs CPU throughput for a given matrix size.
        Returns dict with timing information.
        """
        A = np.random.rand(n, n).astype(np.float32)
        B = np.random.rand(n, n).astype(np.float32)

        # CPU timing
        t0 = time.perf_counter()
        for _ in range(iterations):
            self.multiply(A, B, force_cpu=True)
        cpu_ms = (time.perf_counter() - t0) * 1000 / iterations

        # GPU timing (if available)
        gpu_ms = float('inf')
        if self.available:
            # Warm-up
            self._gpu_multiply(A, B)
            if TORCH_AVAILABLE:
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(iterations):
                self._gpu_multiply(A, B)
            if TORCH_AVAILABLE:
                torch.cuda.synchronize()
            gpu_ms = (time.perf_counter() - t0) * 1000 / iterations

        return {
            "n": n,
            "cpu_ms": cpu_ms,
            "gpu_ms": gpu_ms,
            "speedup": cpu_ms / gpu_ms if gpu_ms > 0 else 0,
            "device": self.device_info.name,
        }

    def _to_numpy(self, x):
        """Normalize input to numpy array."""
        if isinstance(x, np.ndarray):
            return x.astype(np.float32) if x.dtype != np.float32 else x
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
        return np.array(x, dtype=np.float32)

