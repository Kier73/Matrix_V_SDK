"""
Integration Test: RNS-Backed SDK Dispatch
==========================================
Verifies that the RNS-extended signature is properly integrated
into the main SDK dispatch path.

Run: python tests/integration/test_rns_integration.py
"""

import sys
import os
import time
import struct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from matrix_v_sdk.vl.substrate.matrix import SymbolicDescriptor, InfiniteMatrix, MatrixOmega
from matrix_v_sdk.vl.substrate.rns_signature import RNSSignature, RNS_PRIMES
from matrix_v_sdk.vl.substrate.bounded import (
    BoundedDescriptor, ProcessShape, LogicShape, bounded_from_seed, bounded_matmul
)

PASS = 0
FAIL = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


# ==============================================================
# 1. SymbolicDescriptor is RNS-backed
# ==============================================================

def test_symbolic_rns_backed():
    print("\n" + "=" * 60)
    print("1. SymbolicDescriptor is RNS-backed")
    print("=" * 60)

    A = SymbolicDescriptor(100, 100, 0xAAAA)
    B = SymbolicDescriptor(100, 100, 0xBBBB)

    # Has residues
    check("has .residues", hasattr(A, 'residues'))
    check("residues is tuple", isinstance(A.residues, tuple))
    check("len(residues) == 8", len(A.residues) == len(RNS_PRIMES))

    # Has signature (backward compat)
    check("has .signature", hasattr(A, 'signature'))
    check("signature is int", isinstance(A.signature, int))

    # Can compose
    C = A.multiply(B)
    check("multiply returns SymbolicDescriptor",
          isinstance(C, SymbolicDescriptor))
    check("C has residues", hasattr(C, 'residues'))
    check("C rows correct", C.rows == 100)
    check("C cols correct", C.cols == 100)

    # Associativity
    D = SymbolicDescriptor(100, 100, 0xCCCC)
    left = A.multiply(B).multiply(D)
    right = A.multiply(B.multiply(D))
    check("associative: (A*B)*C == A*(B*C)",
          left.residues == right.residues)

    # Identity
    I = SymbolicDescriptor.identity(100, 100)
    AI = A.multiply(I)
    check("identity: A*I == A", AI.residues == A.residues)

    # Resolve
    v = A.resolve(50, 50)
    check("resolve returns float", isinstance(v, float))

    # Resolve reproducibility
    v2 = A.resolve(50, 50)
    check("resolve reproducible",
          struct.pack('d', v) == struct.pack('d', v2))

    # Add and scale (new API)
    S = A.add(B)
    check("add returns SymbolicDescriptor", isinstance(S, SymbolicDescriptor))
    Sc = A.scale(7)
    check("scale returns SymbolicDescriptor", isinstance(Sc, SymbolicDescriptor))


# ==============================================================
# 2. MatrixOmega dispatches through RNS
# ==============================================================

def test_omega_dispatch():
    print("\n" + "=" * 60)
    print("2. MatrixOmega dispatches through RNS")
    print("=" * 60)

    omega = MatrixOmega()
    A = SymbolicDescriptor(100, 100, 0xAAAA)
    B = SymbolicDescriptor(100, 100, 0xBBBB)

    # auto_select_strategy should return "symbolic"
    strategy = omega.auto_select_strategy(A, B)
    check("strategy is symbolic", strategy == "symbolic")

    # resolve_symbolic should use RNS path
    result = omega.resolve_symbolic(A, B)
    check("resolve_symbolic returns object with residues",
          hasattr(result, 'residues'))
    check("result has correct dims",
          result.rows == 100 and result.cols == 100)

    # compute_product should work
    result2 = omega.compute_product(A, B)
    check("compute_product returns structured result",
          result2 is not None)


# ==============================================================
# 3. InfiniteMatrix with RNSSignature
# ==============================================================

def test_infinite_matrix():
    print("\n" + "=" * 60)
    print("3. InfiniteMatrix with RNSSignature")
    print("=" * 60)

    # From SymbolicDescriptor
    desc = SymbolicDescriptor(1000, 1000, 42)
    m1 = InfiniteMatrix(desc)
    check("InfiniteMatrix from SymbolicDescriptor", m1.shape == (1000, 1000))
    check("element access works", isinstance(m1[500, 500], float))

    # From RNSSignature directly
    rns = RNSSignature(1000, 1000, 42)
    m2 = InfiniteMatrix(rns)
    check("InfiniteMatrix from RNSSignature", m2.shape == (1000, 1000))
    check("RNS element access works", isinstance(m2[500, 500], float))

    # Matmul
    m3 = m1.matmul(m2)
    check("matmul returns InfiniteMatrix", isinstance(m3, InfiniteMatrix))
    check("matmul dims correct", m3.shape == (1000, 1000))

    # Matmul associativity
    d1 = SymbolicDescriptor(100, 100, 1)
    d2 = SymbolicDescriptor(100, 100, 2)
    d3 = SymbolicDescriptor(100, 100, 3)
    im1 = InfiniteMatrix(d1)
    im2 = InfiniteMatrix(d2)
    im3 = InfiniteMatrix(d3)
    left = im1.matmul(im2).matmul(im3)
    right = im1.matmul(im2.matmul(im3))
    check("InfiniteMatrix matmul is associative",
          left.desc.residues == right.desc.residues)


# ==============================================================
# 4. BoundedDescriptor with RNS backend
# ==============================================================

def test_bounded_rns():
    print("\n" + "=" * 60)
    print("4. BoundedDescriptor with RNS backend")
    print("=" * 60)

    A = bounded_from_seed(100, 100, seed=42, value_bound=1.0)
    B = bounded_from_seed(100, 100, seed=99, value_bound=1.0)

    check("has .residues", hasattr(A, 'residues'))
    check("has .signature", hasattr(A, 'signature'))
    check("residues is tuple", isinstance(A.residues, tuple))

    # Compose
    C = bounded_matmul(A, B)
    check("bounded_matmul works", isinstance(C, BoundedDescriptor))
    check("C has residues", hasattr(C, 'residues'))

    # Resolve
    v = C.resolve(50, 50)
    check("resolve returns float", isinstance(v, float))
    check("value within bound", abs(v) <= C.process.value_bound)

    # Resolve checked
    v2, excl = C.resolve_checked(50, 50)
    ok, viols = excl.verify()
    check("resolve_checked passes", ok)

    # Error exclusion still works
    try:
        C.resolve(150, 50)
        check("OOB exclusion", False)
    except IndexError:
        check("OOB exclusion", True)

    # Dim mismatch exclusion
    X = bounded_from_seed(100, 200, seed=1)
    Y = bounded_from_seed(300, 400, seed=2)
    try:
        bounded_matmul(X, Y)
        check("dim mismatch exclusion", False)
    except ValueError:
        check("dim mismatch exclusion", True)

    # Chain 100 at scale
    t0 = time.perf_counter()
    result = A
    for _ in range(99):
        result = bounded_matmul(result, A)
    t_chain = (time.perf_counter() - t0) * 1000
    check(f"100-chain in {t_chain:.1f}ms", t_chain < 5000)
    v_chain = result.resolve(50, 50)
    check("chain resolve works", isinstance(v_chain, float))
    check("chain value finite", not (abs(v_chain) == float('inf')))


# ==============================================================
# 5. Top-level imports work
# ==============================================================

def test_top_level_imports():
    print("\n" + "=" * 60)
    print("5. Top-level imports from matrix_v_sdk.vl")
    print("=" * 60)

    from matrix_v_sdk.vl import RNSSignature as RS
    from matrix_v_sdk.vl import BoundedDescriptor as BD
    from matrix_v_sdk.vl import MemthematicPipeline as MP
    from matrix_v_sdk.vl import SymbolicDescriptor as SD
    from matrix_v_sdk.vl import InfiniteMatrix as IM
    from matrix_v_sdk.vl import fmix64 as fm

    check("import RNSSignature", RS is not None)
    check("import BoundedDescriptor", BD is not None)
    check("import MemthematicPipeline", MP is not None)
    check("import SymbolicDescriptor", SD is not None)
    check("import InfiniteMatrix", IM is not None)
    check("import fmix64", fm is not None)

    # Quick functional check
    sig = RS(100, 100, 42)
    desc = SD(100, 100, 42)
    check("RNSSignature functional", sig.resolve(0, 0) is not None)
    check("SymbolicDescriptor functional", desc.resolve(0, 0) is not None)


# ==============================================================
# 6. Backward compatibility
# ==============================================================

def test_backward_compat():
    print("\n" + "=" * 60)
    print("6. Backward compatibility")
    print("=" * 60)

    # Old-style construction
    old = SymbolicDescriptor(100, 100, 0xDEAD, depth=1)
    check("old-style construction works", old is not None)
    check("has rows", old.rows == 100)
    check("has cols", old.cols == 100)
    check("has depth", old.depth == 1)
    check("has signature (int)", isinstance(old.signature, int))

    # Old-style multiply
    old2 = SymbolicDescriptor(100, 100, 0xBEEF, depth=1)
    result = old.multiply(old2)
    check("old-style multiply works", result is not None)
    check("result has rows", result.rows == 100)
    check("result depth incremented", result.depth == 2)

    # Old-style resolve
    v = old.resolve(0, 0)
    check("old-style resolve works", isinstance(v, float))
    check("value in [-1, 1]", -1.0 <= v <= 1.0)

    # Signature setter (legacy)
    old.signature = 0xCAFE
    v2 = old.resolve(0, 0)
    check("signature setter works", isinstance(v2, float))

    # InfiniteMatrix old-style
    im = InfiniteMatrix(old)
    check("InfiniteMatrix old-style", im[0, 0] is not None)
    im2 = InfiniteMatrix(old2)
    result = im.matmul(im2)
    check("InfiniteMatrix matmul old-style", result is not None)


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    print("+" + "=" * 58 + "+")
    print("|  RNS-BACKED SDK INTEGRATION TEST                         |")
    print("+" + "=" * 58 + "+")

    t0 = time.perf_counter()

    test_symbolic_rns_backed()
    test_omega_dispatch()
    test_infinite_matrix()
    test_bounded_rns()
    test_top_level_imports()
    test_backward_compat()

    total = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(f"RESULT: {PASS} passed, {FAIL} failed  ({total:.2f}s)")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)


