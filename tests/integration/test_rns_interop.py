"""
RNS Signature Interoperability with External Frameworks
=========================================================
Tests that the RNS-backed symbolic descriptors interoperate cleanly
with the existing bridge infrastructure (PyTorch, NumPy, SciPy, etc.).

Demonstrates:
  1. RNS -> Tensor:  Materialize symbolic descriptor regions into framework tensors
  2. Tensor -> RNS:  Capture tensor content as RNS ledger for verification
  3. RNS -> PyTorch:  Materialize and use in torch operations
  4. Bridge Dispatch: MatrixOmega routes SymbolicDescriptor through RNS
  5. Bounded -> NumPy: Materialize bounded descriptors with error exclusion
  6. RNS Ledger:      Verify materialized tensors against RNS fingerprints
  7. InfiniteMatrix:   Slice trillion-scale matrices into NumPy
  8. PyTorch -> RNS:   Roundtrip verification
  9. Full chain:       Bridge dispatch at exascale

Run: python tests/integration/test_rns_interop.py
"""

import sys
import os
import time
import struct
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from matrix_v_sdk.vl.substrate.matrix import SymbolicDescriptor, InfiniteMatrix, MatrixOmega
from matrix_v_sdk.vl.substrate.rns_signature import RNSSignature, RNS_PRIMES
from matrix_v_sdk.vl.substrate.bounded import (
    BoundedDescriptor, ProcessShape, LogicShape,
    bounded_from_seed, bounded_matmul,
)
from matrix_v_sdk.vl.substrate.rns_ledger import RNSLedger, MODULI, record_matrix, verify_matrix

# Optional imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

PASS = 0
FAIL = 0
SKIP = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")


def skip(name, reason):
    global SKIP
    SKIP += 1
    print(f"  [SKIP] {name} -- {reason}")


# ==============================================================
# 1. RNS -> NumPy Materialization
# ==============================================================

def test_rns_to_numpy():
    print("\n" + "=" * 60)
    print("1. RNS -> NumPy Materialization")
    print("=" * 60)

    sig = RNSSignature(1000, 1000, 0xCAFE)

    # Materialize a 100x100 sub-block
    t0 = time.perf_counter()
    rows, cols = 100, 100
    arr = np.array([
        [sig.resolve(r, c) for c in range(cols)]
        for r in range(rows)
    ])
    t_mat = (time.perf_counter() - t0) * 1000

    check("shape correct", arr.shape == (100, 100))
    check("values in [-1, 1]", np.all(np.abs(arr) <= 1.0))
    check("no NaN", not np.any(np.isnan(arr)))
    check("no Inf", not np.any(np.isinf(arr)))
    check("non-trivial", np.any(arr != 0.0))

    mean = np.mean(arr)
    std = np.std(arr)
    check(f"mean near 0 (got {mean:.4f})", abs(mean) < 0.1)
    check(f"std > 0.3 (got {std:.4f})", std > 0.3)

    # Reproducibility
    arr2 = np.array([
        [sig.resolve(r, c) for c in range(cols)]
        for r in range(rows)
    ])
    check("bit-exact reproducible", np.array_equal(arr, arr2))

    print(f"\n  Materialized 100x100 = 10,000 elements in {t_mat:.1f}ms")


# ==============================================================
# 2. NumPy -> RNS Ledger Verification
# ==============================================================

def test_numpy_to_rns_ledger():
    print("\n" + "=" * 60)
    print("2. NumPy -> RNS Ledger Verification")
    print("=" * 60)

    np.random.seed(42)
    A = np.random.randn(8, 8)

    # Record into RNS ledger
    ledger = record_matrix(A.tolist())
    check("ledger created", ledger is not None)
    check("ledger has entries", len(ledger._entries) == 64)

    # Verify same data
    ok, passes, total = verify_matrix(A.tolist(), ledger)
    check(f"self-verification: {passes}/{total}", ok)

    # Verify tampered data fails
    B = A.copy()
    B[3, 4] += 1e-4  # tiny perturbation
    ok2, passes2, total2 = verify_matrix(B.tolist(), ledger)
    check("tamper detected", not ok2)
    check(f"tamper isolated: {passes2}/{total2}", passes2 == 63)

    print(f"\n  Matrix:  8x8 = 64 elements")
    print(f"  Ledger:  {ledger.stats()}")


# ==============================================================
# 3. RNS -> PyTorch Tensor
# ==============================================================

def test_rns_to_torch():
    print("\n" + "=" * 60)
    print("3. RNS -> PyTorch Tensor")
    print("=" * 60)

    if not HAS_TORCH:
        skip("PyTorch interop", "torch not installed")
        return

    desc = SymbolicDescriptor(500, 500, 42)

    # Materialize 50x50 as torch tensor
    t0 = time.perf_counter()
    data = [[desc.resolve(r, c) for c in range(50)] for r in range(50)]
    tensor = torch.tensor(data, dtype=torch.float64)
    t_mat = (time.perf_counter() - t0) * 1000

    check("tensor shape", tensor.shape == (50, 50))
    check("tensor dtype", tensor.dtype == torch.float64)
    check("values bounded", torch.all(torch.abs(tensor) <= 1.0).item())
    check("no NaN", not torch.any(torch.isnan(tensor)).item())

    # PyTorch matmul
    C_torch = torch.mm(tensor, tensor)
    check("torch.mm works on materialized data", C_torch.shape == (50, 50))
    check("result finite", torch.all(torch.isfinite(C_torch)).item())

    # Symbolic compose still works
    desc2 = desc.multiply(desc)
    check("symbolic compose works", desc2.rows == 500 and desc2.cols == 500)

    print(f"\n  Materialized 50x50 into torch.Tensor in {t_mat:.1f}ms")
    print(f"  torch.mm result range: [{C_torch.min():.4f}, {C_torch.max():.4f}]")


# ==============================================================
# 4. MatrixOmega Dispatch: Symbolic vs Dense
# ==============================================================

def test_omega_dispatch():
    print("\n" + "=" * 60)
    print("4. MatrixOmega Dispatch: Symbolic vs Dense")
    print("=" * 60)

    omega = MatrixOmega()

    # Symbolic path
    A = SymbolicDescriptor(1000, 1000, 0xAAAA)
    B = SymbolicDescriptor(1000, 1000, 0xBBBB)

    t0 = time.perf_counter()
    strategy = omega.auto_select_strategy(A, B)
    result = omega.resolve_symbolic(A, B)
    t_sym = (time.perf_counter() - t0) * 1e6

    check("strategy is symbolic", strategy == "symbolic")
    check("result has .residues", hasattr(result, 'residues'))
    check("result dims correct", result.rows == 1000 and result.cols == 1000)

    vals = [result.resolve(i, i) for i in range(10)]
    check("composed resolve works", all(isinstance(v, float) for v in vals))

    # Dense path for comparison
    n = 16
    A_dense = np.random.randn(n, n).tolist()
    B_dense = np.random.randn(n, n).tolist()
    strat_dense = omega.auto_select_strategy(A_dense, B_dense)
    C_dense = omega.compute_product(A_dense, B_dense)

    expected = (np.array(A_dense) @ np.array(B_dense)).tolist()
    max_err = max(abs(C_dense[i][j] - expected[i][j])
                  for i in range(n) for j in range(n))
    check(f"dense result correct (err={max_err:.2e})", max_err < 1e-8)

    print(f"\n  Symbolic: {t_sym:.1f}us (1000x1000)")
    print(f"  Both paths through MatrixOmega dispatch.")


# ==============================================================
# 5. BoundedDescriptor -> NumPy
# ==============================================================

def test_bounded_to_numpy():
    print("\n" + "=" * 60)
    print("5. BoundedDescriptor -> NumPy")
    print("=" * 60)

    bound = 5.0
    A = bounded_from_seed(10000, 10000, seed=42, value_bound=bound)
    B = bounded_from_seed(10000, 10000, seed=99, value_bound=bound)
    C = bounded_matmul(A, B)

    # Materialize 50x50
    t0 = time.perf_counter()
    block = np.array([
        [C.resolve(r, c) for c in range(50)]
        for r in range(50)
    ])
    t_mat = (time.perf_counter() - t0) * 1000

    check("shape correct", block.shape == (50, 50))
    check("all within bound", np.all(np.abs(block) <= bound))
    check("no NaN", not np.any(np.isnan(block)))
    check("no Inf", not np.any(np.isinf(block)))
    check("non-trivial", np.std(block) > 0.1)
    check("C has .residues", isinstance(C.residues, tuple))

    print(f"\n  Materialized 50x50 from 10Kx10K in {t_mat:.1f}ms")
    print(f"  Bound: [-{bound}, +{bound}], actual: [{block.min():.4f}, {block.max():.4f}]")


# ==============================================================
# 6. RNS Ledger Verification of Materialized Data
# ==============================================================

def test_rns_ledger_verification():
    print("\n" + "=" * 60)
    print("6. Materialize -> Record -> Verify Integrity")
    print("=" * 60)

    desc = RNSSignature(100, 100, 777)
    n = 20

    # Materialize
    block = [[desc.resolve(r, c) for c in range(n)] for r in range(n)]

    # Record
    ledger = record_matrix(block)

    # Verify
    ok, passes, total = verify_matrix(block, ledger)
    check(f"self-verify: {passes}/{total}", ok)

    # Re-materialize and verify
    block2 = [[desc.resolve(r, c) for c in range(n)] for r in range(n)]
    ok2, passes2, total2 = verify_matrix(block2, ledger)
    check(f"re-materialize verify: {passes2}/{total2}", ok2)

    # Tamper detection
    block_t = [row[:] for row in block]
    block_t[10][10] += 1e-4
    ok3, p3, t3 = verify_matrix(block_t, ledger)
    check("tamper detected", not ok3)

    print(f"\n  Chain: RNSSignature -> list -> RNSLedger -> verify")


# ==============================================================
# 7. InfiniteMatrix -> NumPy slice
# ==============================================================

def test_infinite_matrix_slice():
    print("\n" + "=" * 60)
    print("7. InfiniteMatrix -> NumPy Slice (1T scale)")
    print("=" * 60)

    N = 1_000_000_000_000
    im = InfiniteMatrix(SymbolicDescriptor(N, N, 42))
    check("shape is 1T x 1T", im.shape == (N, N))

    # Slice from center
    center = N // 2
    t0 = time.perf_counter()
    window = np.array([
        [im[center + r, center + c] for c in range(10)]
        for r in range(10)
    ])
    t_slice = (time.perf_counter() - t0) * 1000

    check("window shape", window.shape == (10, 10))
    check("values finite", np.all(np.isfinite(window)))
    check("values bounded", np.all(np.abs(window) <= 1.0))

    # Compose two 1T matrices then slice
    im2 = InfiniteMatrix(SymbolicDescriptor(N, N, 99))
    t0 = time.perf_counter()
    im3 = im.matmul(im2)
    t_compose = (time.perf_counter() - t0) * 1e6

    product_window = np.array([
        [im3[r, c] for c in range(10)]
        for r in range(10)
    ])
    check("product slice works", product_window.shape == (10, 10))
    check("product finite", np.all(np.isfinite(product_window)))

    # Associativity via numpy
    im4 = InfiniteMatrix(SymbolicDescriptor(N, N, 7))
    left = im.matmul(im2).matmul(im4)
    right = im.matmul(im2.matmul(im4))
    check("associative through numpy", left.desc.residues == right.desc.residues)

    print(f"\n  Source: {N:,}x{N:,}, slice: 10x10 in {t_slice:.1f}ms")
    print(f"  Compose: {t_compose:.1f}us")


# ==============================================================
# 8. PyTorch -> RNS Roundtrip
# ==============================================================

def test_torch_rns_roundtrip():
    print("\n" + "=" * 60)
    print("8. PyTorch -> RNS Roundtrip")
    print("=" * 60)

    if not HAS_TORCH:
        skip("PyTorch roundtrip", "torch not installed")
        return

    torch.manual_seed(42)
    T = torch.randn(16, 16, dtype=torch.float64)

    # Record tensor in ledger
    ledger = record_matrix(T.tolist())

    # Verify: same tensor
    ok, passes, total = verify_matrix(T.tolist(), ledger)
    check(f"torch tensor self-verify: {passes}/{total}", ok)

    # Roundtrip: torch -> numpy -> torch -> verify
    T_np = T.numpy()
    T2 = torch.from_numpy(T_np.copy())
    ok2, p2, t2 = verify_matrix(T2.tolist(), ledger)
    check(f"torch->numpy->torch roundtrip: {p2}/{t2}", ok2)

    # Roundtrip: torch -> list -> torch -> verify
    T3 = torch.tensor(T.tolist(), dtype=torch.float64)
    ok3, p3, t3 = verify_matrix(T3.tolist(), ledger)
    check(f"torch->list->torch roundtrip: {p3}/{t3}", ok3)

    # Matmul fingerprint determinism
    C1 = torch.mm(T, T)
    C2 = torch.mm(T, T)
    ledger_c1 = record_matrix(C1.tolist())
    ok4, p4, t4 = verify_matrix(C2.tolist(), ledger_c1)
    check(f"matmul fingerprint deterministic: {p4}/{t4}", ok4)

    print(f"\n  Roundtrip chain: torch -> list -> RNSLedger -> verify")


# ==============================================================
# 9. Full Bridge Chain
# ==============================================================

def test_bridge_chain():
    print("\n" + "=" * 60)
    print("9. Full Bridge Chain: Symbolic -> Omega -> NumPy -> Verify")
    print("=" * 60)

    omega = MatrixOmega()
    N = 1_000_000_000

    A = SymbolicDescriptor(N, N, 42)

    t0 = time.perf_counter()
    result = A
    for i in range(9):
        B = SymbolicDescriptor(N, N, i + 100)
        result = omega.resolve_symbolic(result, B)
    t_chain = (time.perf_counter() - t0) * 1000

    check("chain has residues", hasattr(result, 'residues'))
    check("chain depth == 10", result.depth == 10)
    check("chain dims", result.rows == N and result.cols == N)

    # Materialize 10 diagonal elements
    vals = [result.resolve(i * 1000, i * 1000) for i in range(10)]
    check("all finite", all(math.isfinite(v) for v in vals))

    # Record into ledger (as 1x10)
    ledger = RNSLedger(1, 10)
    for i, v in enumerate(vals):
        ledger.record(0, i, v)

    # Re-materialize and verify
    vals2 = [result.resolve(i * 1000, i * 1000) for i in range(10)]
    all_verified = all(ledger.verify_residues(0, i, vals2[i]) for i in range(10))
    check("re-materialized values verify", all_verified)

    print(f"\n  Chain: 10 x {N:,}x{N:,} in {t_chain:.1f}ms")
    print(f"  Result residues: {result.residues}")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    print("+" + "=" * 58 + "+")
    print("|  RNS INTEROPERABILITY WITH EXTERNAL FRAMEWORKS            |")
    print("+" + "=" * 58 + "+")

    t_total = time.perf_counter()

    test_rns_to_numpy()
    test_numpy_to_rns_ledger()
    test_rns_to_torch()
    test_omega_dispatch()
    test_bounded_to_numpy()
    test_rns_ledger_verification()
    test_infinite_matrix_slice()
    test_torch_rns_roundtrip()
    test_bridge_chain()

    total = time.perf_counter() - t_total

    print("\n" + "=" * 60)
    print(f"RESULT: {PASS} passed, {FAIL} failed, {SKIP} skipped  "
          f"({total:.2f}s)")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)


