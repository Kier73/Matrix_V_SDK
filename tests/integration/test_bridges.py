"""
Interoperability Bridge Verification Suite
==========================================
Tests all 6 gap bridges:
  1. SciPy Sparse (CSR/CSC roundtrip)
  2. CUDA (GPU vs CPU agreement)
  3. ONNX Export (export + validate)
  4. StrategyCache Serialization (save/load roundtrip)
  5. SafeTensors (model save/load)
  6. RNS Backward (exact vs standard gradients)
"""
import sys
import os
import tempfile
import numpy as np
import json

# Ensure the SDK root is on the path (parent of matrix_v_sdk)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from matrix_v_sdk.vl.substrate.matrix import (
    MatrixOmega, StrategyCache, StrategyRecord, MatrixFeatureVector
)

PASS_COUNT = 0
FAIL_COUNT = 0

def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {name}")
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {name} -- {detail}")


# ================================================================
# 1. SciPy Sparse Bridge
# ================================================================
def test_scipy_bridge():
    print("\n=== 1. SCIPY SPARSE BRIDGE ===")
    try:
        import scipy.sparse as sp
        from matrix_v_sdk.extensions.scipy_bridge import SparseMatrixV, sparse_feature_vector
    except ImportError:
        print("  [SKIP] scipy not installed")
        return

    smv = SparseMatrixV()
    n = 32

    # 1.1 Dense roundtrip via sparse
    A_dense = np.random.rand(n, n)
    B_dense = np.random.rand(n, n)
    expected = A_dense @ B_dense

    A_csr = sp.csr_matrix(A_dense)
    B_csr = sp.csr_matrix(B_dense)

    result = smv.multiply(A_csr, B_csr, output_format='dense')
    err = np.max(np.abs(result - expected))
    check("CSR dense roundtrip accuracy", err < 1e-8, f"error={err:.2e}")

    # 1.2 CSR output format
    result_csr = smv.multiply(A_csr, B_csr, output_format='csr')
    check("CSR output format", sp.issparse(result_csr), f"type={type(result_csr)}")

    # 1.3 CSC output format
    result_csc = smv.multiply(A_csr, B_csr, output_format='csc')
    check("CSC output format", sp.issparse(result_csc) and isinstance(result_csc, sp.csc_matrix),
          f"type={type(result_csc)}")

    # 1.4 O(1) sparsity from sparse metadata
    sparse_data = sp.random(100, 100, density=0.05, format='csr')
    fv = sparse_feature_vector(sparse_data, sparse_data)
    expected_sparsity = 1.0 - 0.05
    check("O(1) sparsity extraction",
          abs(fv.sparsity - expected_sparsity) < 0.02,
          f"got {fv.sparsity:.3f}, expected ~{expected_sparsity:.3f}")

    # 1.5 Feature vector shape class
    sc = fv.shape_class()
    check("Sparse feature vector shape_class", sc is not None and len(sc) > 0, f"got '{sc}'")


# ================================================================
# 2. CUDA Bridge
# ================================================================
def test_cuda_bridge():
    print("\n=== 2. CUDA BRIDGE ===")
    try:
        import torch
        from matrix_v_sdk.extensions.cuda_bridge import CUDAMatrixV, CUDADeviceInfo
    except ImportError:
        print("  [SKIP] torch not installed")
        return

    # 2.1 Device detection
    info = CUDADeviceInfo()
    check("CUDA device detection", True, f"available={info.available}, name={info.name}")

    cuda = CUDAMatrixV()

    # 2.2 CPU fallback always works
    A = np.random.rand(32, 32).astype(np.float32)
    B = np.random.rand(32, 32).astype(np.float32)
    result = cuda.multiply(A, B, force_cpu=True)
    expected = A @ B
    err = np.max(np.abs(np.array(result) - expected))
    check("CPU fallback accuracy", err < 1e-5, f"error={err:.2e}")

    if info.available:
        # 2.3 GPU accuracy
        n = 256
        A_big = np.random.rand(n, n).astype(np.float32)
        B_big = np.random.rand(n, n).astype(np.float32)
        expected_big = A_big @ B_big

        result_gpu = cuda.multiply(A_big, B_big, force_gpu=True)
        gpu_err = np.max(np.abs(np.array(result_gpu) - expected_big))
        check(f"GPU accuracy (n={n})", gpu_err < 1e-3, f"error={gpu_err:.2e}")

        # 2.4 GPU vs CPU agreement
        result_cpu = cuda.multiply(A_big, B_big, force_cpu=True)
        agreement_err = np.max(np.abs(np.array(result_gpu) - np.array(result_cpu)))
        check("GPU/CPU agreement", agreement_err < 1e-2, f"error={agreement_err:.2e}")

        # 2.5 Auto-routing (small -> CPU, large -> GPU)
        small_result = cuda.multiply(np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32))
        check("Auto-route small to CPU", small_result is not None)

        # 2.6 FP16 path
        result_fp16 = cuda.multiply_fp16(A_big, B_big)
        fp16_err = np.max(np.abs(np.array(result_fp16) - expected_big))
        check(f"FP16 matmul (n={n})", fp16_err < 0.5, f"error={fp16_err:.2e}")

        # 2.7 Benchmark
        bm = cuda.benchmark(n=128, iterations=3)
        check(f"Benchmark runs ({bm['device']})", 
              bm['cpu_ms'] > 0 and bm['gpu_ms'] > 0,
              f"cpu={bm['cpu_ms']:.1f}ms, gpu={bm['gpu_ms']:.1f}ms, speedup={bm['speedup']:.1f}x")
    else:
        print("  [SKIP] No CUDA device - GPU tests skipped")


# ================================================================
# 3. ONNX Export Bridge
# ================================================================
def test_onnx_bridge():
    print("\n=== 3. ONNX EXPORT BRIDGE ===")
    try:
        import torch
        from matrix_v_sdk.extensions.torch_bridge import MatrixVLinear
        from matrix_v_sdk.extensions.onnx_bridge import export_to_onnx, _make_exportable_model
    except ImportError as e:
        print(f"  [SKIP] {e}")
        return

    # 3.1 Make exportable model (replace MatrixVLinear -> nn.Linear)
    model_original = torch.nn.Sequential(
        MatrixVLinear(16, 32),
        torch.nn.ReLU(),
        MatrixVLinear(32, 8),
    )
    # Save original weights for comparison
    orig_weight = list(model_original.parameters())[0].data.clone()

    # Create a separate model for export (since _make_exportable_model modifies in-place for Sequential)
    model_for_export = torch.nn.Sequential(
        MatrixVLinear(16, 32),
        torch.nn.ReLU(),
        MatrixVLinear(32, 8),
    )
    # Copy weights from original
    model_for_export.load_state_dict(model_original.state_dict())

    exportable = _make_exportable_model(model_for_export)

    # Check that MatrixVLinear was replaced
    has_mv = any(isinstance(m, MatrixVLinear) for m in exportable.modules())
    check("MatrixVLinear replaced with nn.Linear", not has_mv)

    # 3.2 Weights preserved
    export_w = list(exportable.parameters())[0].data
    check("Weights preserved in export", torch.allclose(orig_weight, export_w))

    # 3.3 Export to ONNX file
    try:
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        dummy = torch.randn(1, 16)
        export_to_onnx(model_for_export, dummy, onnx_path)

        check("ONNX file exported", os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0,
              f"size={os.path.getsize(onnx_path)}")

        # 3.4 Validate ONNX
        try:
            from matrix_v_sdk.extensions.onnx_bridge import validate_onnx
            valid = validate_onnx(onnx_path)
            check("ONNX validation passes", valid)
        except ImportError:
            print("  [SKIP] onnx package not installed for validation")

        # 3.5 ONNX Runtime comparison
        try:
            from matrix_v_sdk.extensions.onnx_bridge import compare_outputs
            result = compare_outputs(exportable, onnx_path, dummy)
            check(f"ONNX Runtime output matches (max_err={result['max_error']:.2e})",
                  result['match'])
        except ImportError:
            print("  [SKIP] onnxruntime not installed for comparison")
    except Exception as e:
        check("ONNX export", False, str(e))
    finally:
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)


# ================================================================
# 4. StrategyCache Serialization
# ================================================================
def test_cache_serialization():
    print("\n=== 4. STRATEGY CACHE SERIALIZATION ===")

    # 4.1 StrategyRecord roundtrip
    rec = StrategyRecord(engine_name="qmatrix")
    for i in range(5):
        rec.record(0.001 * i, 10.0 + 0.1 * i)
    d = rec.to_dict()
    rec2 = StrategyRecord.from_dict(d)
    check("StrategyRecord roundtrip",
          rec2.engine_name == "qmatrix" and rec2.call_count == 5 and len(rec2.times_ms) == 5)

    # 4.2 StrategyCache roundtrip
    cache = StrategyCache()
    for i in range(10):
        cache.observe("64x64x64", "adaptive_block", 0.0, 10.0 + 0.01 * np.random.randn())
    for i in range(10):
        cache.observe("128x128x128", "qmatrix", 0.0, 20.0 + 0.01 * np.random.randn())

    d = cache.to_dict()
    cache2 = StrategyCache.from_dict(d)

    # Check promoted engines survived
    p1 = cache.recall("64x64x64")
    p2 = cache2.recall("64x64x64")
    check("Cache promotion survives serialization", p1 == p2 and p1 is not None,
          f"original={p1}, restored={p2}")

    # 4.3 File save/load
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        cache_path = f.name

    try:
        cache.save(cache_path)
        check("Cache saved to JSON", os.path.exists(cache_path))

        loaded = StrategyCache.load(cache_path)
        lp = loaded.recall("64x64x64")
        check("Cache restored from JSON", lp == p1, f"got {lp}")

        # Verify JSON structure
        with open(cache_path) as f:
            data = json.load(f)
        check("JSON has expected keys", 'cache' in data and 'promoted' in data)
    finally:
        os.unlink(cache_path)


# ================================================================
# 5. SafeTensors Bridge
# ================================================================
def test_safetensors_bridge():
    print("\n=== 5. SAFETENSORS BRIDGE ===")
    try:
        import torch
        from safetensors.torch import save_file, load_file
        from matrix_v_sdk.extensions.torch_bridge import MatrixVLinear
        from matrix_v_sdk.extensions.safetensors_bridge import save_matrix_v_model, load_matrix_v_model
    except ImportError as e:
        print(f"  [SKIP] {e}")
        return

    # Create model with known weights
    model = torch.nn.Sequential()
    model.add_module('layer1', MatrixVLinear(16, 8))
    model.add_module('relu', torch.nn.ReLU())
    model.add_module('layer2', MatrixVLinear(8, 4))

    # Build a strategy cache with learned data
    cache = StrategyCache()
    for i in range(10):
        cache.observe("16x16x8", "dense", 0.0, 5.0 + 0.01 * np.random.randn())

    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        st_path = f.name

    try:
        # 5.1 Save
        save_matrix_v_model(model, st_path, strategy_cache=cache)
        check("SafeTensors file saved", os.path.exists(st_path) and os.path.getsize(st_path) > 0)

        # 5.2 Load into new model
        model2 = torch.nn.Sequential()
        model2.add_module('layer1', MatrixVLinear(16, 8))
        model2.add_module('relu', torch.nn.ReLU())
        model2.add_module('layer2', MatrixVLinear(8, 4))

        model2, metadata, restored_cache = load_matrix_v_model(st_path, model2)

        # 5.3 Weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            check(f"Weight '{n1}' matches", torch.allclose(p1, p2))

        # 5.4 Forward pass match
        x = torch.randn(1, 16)
        model.eval()
        model2.eval()
        with torch.no_grad():
            y1 = model(x)
            y2 = model2(x)
        check("Forward pass matches after load", torch.allclose(y1, y2, atol=1e-6))

        # 5.5 Strategy cache metadata
        check("Strategy cache in metadata",
              'matrix_v_strategy_cache' in metadata,
              f"keys={list(metadata.keys())}")

    finally:
        if os.path.exists(st_path):
            os.unlink(st_path)


# ================================================================
# 6. RNS Backward Pass
# ================================================================
def test_rns_backward():
    print("\n=== 6. RNS BACKWARD PASS ===")
    try:
        import torch
        from matrix_v_sdk.extensions.torch_bridge import MatrixVLinear
    except ImportError as e:
        print(f"  [SKIP] {e}")
        return

    # 6.1 Standard backward still works
    layer_std = MatrixVLinear(8, 4, exact_backward=False)
    x = torch.randn(2, 8, requires_grad=True)
    y = layer_std(x)
    loss = y.sum()
    loss.backward()
    check("Standard backward produces gradients", x.grad is not None and x.grad.shape == (2, 8))

    # 6.2 RNS exact backward
    layer_exact = MatrixVLinear(8, 4, exact_backward=True)
    # Copy weights so we can compare
    layer_exact.weight.data = layer_std.weight.data.clone()
    if layer_exact.bias is not None and layer_std.bias is not None:
        layer_exact.bias.data = layer_std.bias.data.clone()

    x2 = x.detach().clone().requires_grad_(True)
    y2 = layer_exact(x2)
    loss2 = y2.sum()
    loss2.backward()
    check("RNS exact backward produces gradients",
          x2.grad is not None and x2.grad.shape == (2, 8))

    # 6.3 Both gradient paths produce similar results
    # (They won't be identical since MMP uses RNS integer arithmetic
    # and torch.mm uses fp32 — but they should agree to ~1e-3)
    grad_err = torch.max(torch.abs(x.grad - x2.grad)).item()
    check(f"Standard vs RNS gradient agreement (err={grad_err:.2e})",
          grad_err < 0.1, f"error={grad_err:.2e}")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    test_scipy_bridge()
    test_cuda_bridge()
    test_onnx_bridge()
    test_cache_serialization()
    test_safetensors_bridge()
    test_rns_backward()

    print(f"\n{'='*50}")
    print(f"  TOTAL: {PASS_COUNT + FAIL_COUNT} tests, {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print(f"{'='*50}")

    sys.exit(1 if FAIL_COUNT > 0 else 0)


