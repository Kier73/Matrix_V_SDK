# MMP-Series: Modular Manifold Engine

The MMP-Series is the **Operational Core** of the Residue Number System (RNS) substrate. It handles the parallel residue channels that enable high-precision, exact-parity matrix operations for massive integers.

## Theory of Operation

MMP converts standard integer arithmetic into a series of parallel, smaller modular operations across multiple "Prime Channels."

### 1. Residue Decomposition
A large value $X$ is sharded into residues:
$$X \to \{r_1, r_2, \dots, r_n\} \text{ where } r_i = X \pmod{p_i}$$

### 2. Channel Matmul
Matrix multiplication is performed independently in each prime channel. Since the numbers remain small (fitting in 16-bit or 32-bit registers), this is extremely fast and avoids overflow entirely within the 310-bit dynamic range.

### 3. CRT Reconstruction
The final result is "lifted" back into the unified manifold using the Chinese Remainder Theorem:
$$X = \left[ \sum_{k=1}^n r_k M_k y_k \right] \pmod M$$

## Usage

### Monolith Version
```python
from matrix_v_monolith import VlAdaptiveRNS

rns = VlAdaptiveRNS(count=16)
residues = rns.decompose(123456789)
original = rns.reconstruct(residues)
```

### Standalone SDK
```python
from matrix_v_sdk.vl.substrate.rns_signature import RNSSignature

# Handle exact high-precision gradients
sig = RNSSignature.from_matrix(data)
```

## Key Benefits
- **Exact Parity**: Zero error accumulation for integer-equivalent operations.
- **Dynamic Range**: Supports up to 310-bit integers (with 16 primes).
- **Parallelizable**: Each residue channel can run on a separate CPU core or GPU thread.
