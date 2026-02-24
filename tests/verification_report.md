# Matrix-V SDK Verification Report

## Executive Summary
- **Total Tests**: 40
- **Passed**: 36
- **Failed/Error**: 4

### Tier1 basic
| Test Script | Status | Notes |
|-------------|--------|-------|
| 01_test_dot.py | PASS |  |
| 02_test_identity.py | PASS |  |
| 03_test_zeros.py | PASS |  |
| 04_test_rect.py | PASS |  |
| 05_test_scalar.py | PASS |  |
| 06_test_p_init.py | PASS |  |
| 07_test_g_init.py | PASS |  |
| 08_test_v_init.py | PASS |  |
| 09_test_x_init.py | PASS |  |
| 10_test_rh_init.py | PASS |  |

### Tier2 general
| Test Script | Status | Notes |
|-------------|--------|-------|
| 01_test_omega.py | PASS |  |
| 02_test_numpy.py | PASS |  |
| 03_test_torch.py | PASS |  |
| 04_test_jax.py | PASS |  |
| 05_test_numba.py | PASS |  |
| 06_test_errors.py | PASS |  |
| 07_test_determinism.py | PASS |  |
| 08_test_precision.py | FAIL | Traceback (most recent call last):   File "C:\Users\kross\Downloads\Matrix 0(n1.5)\matrix_v_sdk\tests\tier2_general\08_test_precision.py", line 22, in <module>     test_precision()     ~~~~~~~~~~~~~~^^   File "C:\Users\kross\Downloads\Matrix 0(n1.5)\matrix_v_sdk\tests\tier2_general\08_test_precision.py", line 18, in test_precision     assert diff < 1e-4            ^^^^^^^^^^^ AssertionError  |
| 09_test_medium_dense.py | PASS |  |
| 10_test_sparse.py | PASS |  |

### Tier3 advanced
| Test Script | Status | Notes |
|-------------|--------|-------|
| 01_test_autograd.py | PASS |  |
| 02_test_attention.py | PASS |  |
| 03_test_jax_grad.py | PASS |  |
| 04_test_stability.py | TIMEOUT | Execution timed out (30s) |
| 05_test_sidechannel.py | FAIL | Traceback (most recent call last):   File "C:\Users\kross\Downloads\Matrix 0(n1.5)\matrix_v_sdk\tests\tier3_advanced\05_test_sidechannel.py", line 18, in <module>     test_sidechannel()     ~~~~~~~~~~~~~~~~^^   File "C:\Users\kross\Downloads\Matrix 0(n1.5)\matrix_v_sdk\tests\tier3_advanced\05_test_sidechannel.py", line 13, in test_sidechannel     assert det.confidence > 0.5            ^^^^^^^^^^^^^^^^^^^^ AssertionError  |
| 06_test_infinite.py | PASS |  |
| 07_test_rh_distribution.py | PASS |  |
| 08_test_cross_binding.py | PASS |  |
| 09_test_rust_stress.py | PASS |  |
| 10_test_mmp_scaling.py | PASS |  |

### Tier4 impossible
| Test Script | Status | Notes |
|-------------|--------|-------|
| 01_test_chaos_n8192.py | PASS |  |
| 02_test_singular.py | PASS |  |
| 03_test_depth_1000.py | PASS |  |
| 04_test_fp16_drift.py | FAIL | Traceback (most recent call last):   File "C:\Users\kross\Downloads\Matrix 0(n1.5)\matrix_v_sdk\tests\tier4_impossible\04_test_fp16_drift.py", line 23, in <module>     test_fp16_drift()     ~~~~~~~~~~~~~~~^^   File "C:\Users\kross\Downloads\Matrix 0(n1.5)\matrix_v_sdk\tests\tier4_impossible\04_test_fp16_drift.py", line 9, in test_fp16_drift     rns = VlAdaptiveRNS(range_max=1000) TypeError: VlAdaptiveRNS.__init__() got an unexpected keyword argument 'range_max'  |
| 05_test_interference.py | PASS |  |
| 06_test_transformer_100.py | PASS |  |
| 07_test_collisions.py | PASS |  |
| 08_test_ffi_concurrency.py | PASS |  |
| 09_test_non_euclidean.py | PASS |  |
| 10_test_min_d_limit.py | PASS |  |

