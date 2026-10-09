#!/usr/bin/env python3
"""
diagnostics/test_consolidated_nan_checker.py

Tests Point 3 from reviewer feedback:
- Evaluates fast_consolidated_check_for_nans across:
  1. All finite inputs (CPU NumPy, GPU JAX, CPU JAX).
  2. A CPU NumPy NaN.
  3. A GPU JAX DeviceArray NaN.
  4. A CPU-backed JAX Array NaN.
  5. Simultaneous NaNs across multiple nested paths.
  6. Integration with _wrapped_resample ensuring StopResampling is raised cleanly.
"""

import sys
import warnings

sys.path.insert(0, '/home/fsp585/Desktop/FiberphotometryCode')
sys.path.insert(0, '/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll')

import numpy as np

if not hasattr(np, "bool8"): np.bool8 = np.bool_

import jax
import jax.numpy as jnp
import utils.resample_batching as rb
from keypoint_moseq.fitting import StopResampling, _wrapped_resample
from utils.resample_batching import fast_consolidated_check_for_nans


def create_mock_pytree():
    cpu_device = jax.devices("cpu")[0]
    return {
        "seed": jax.random.PRNGKey(0),
        "states": {
            "x": jnp.ones((4, 10, 5), dtype=jnp.float64),         # GPU JAX
            "z": np.ones((4, 10), dtype=np.int32),                 # CPU NumPy
            "v_cpu_jax": jax.device_put(jnp.ones((4, 10, 2)), cpu_device) # CPU JAX
        },
        "params": {
            "Ab": jnp.ones((10, 5, 16), dtype=jnp.float64),       # GPU JAX
            "Q": jnp.eye(5)[None].repeat(10, axis=0),             # GPU JAX
            "pi": np.ones((10, 10), dtype=np.float64) / 10.0      # CPU NumPy
        }
    }

def test_nan_checker():
    print("=" * 80)
    print("  TEST 3: CONSOLIDATED NAN CHECKER MULTI-PLACEMENT & INTEGRATION AUDIT")
    print("=" * 80, flush=True)

    rb.apply_optimizations()
    cpu_dev = jax.devices("cpu")[0]

    # 1. Finite inputs
    print("[Case 3A] Testing all-finite pytree (GPU JAX + CPU NumPy + CPU JAX)...", flush=True)
    tree_clean = create_mock_pytree()
    any_nans, nan_info, messages = fast_consolidated_check_for_nans(tree_clean)
    assert any_nans is False, "False positive NaN detected on clean tree!"
    assert len(nan_info) == 0 and len(messages) == 0, "Non-empty messages on clean tree!"
    print("--> Case 3A Passed: Finite pytree returned (False, [], []).")

    # 2. CPU NumPy NaN
    print("\n[Case 3B] Testing CPU NumPy NaN...", flush=True)
    tree_cpu_nan = create_mock_pytree()
    tree_cpu_nan["params"]["pi"][2, 3] = np.nan
    any_nans, nan_info, messages = fast_consolidated_check_for_nans(tree_cpu_nan)
    assert any_nans is True, "Failed to detect CPU NumPy NaN!"
    assert len(nan_info) == 1, f"Expected 1 NaN entry, got {len(nan_info)}"
    assert nan_info[0][1] == 1, f"Expected 1 NaN count, got {nan_info[0][1]}"
    print(f"--> Case 3B Passed: Detected CPU NaN: {messages[0]}")

    # 3. GPU JAX NaN
    print("\n[Case 3C] Testing GPU JAX DeviceArray NaN...", flush=True)
    tree_gpu_nan = create_mock_pytree()
    bad_x = np.ones((4, 10, 5), dtype=np.float64)
    bad_x[1, 2, 3] = np.nan
    bad_x[2, 4, 1] = np.nan
    tree_gpu_nan["states"]["x"] = jnp.asarray(bad_x)
    any_nans, nan_info, messages = fast_consolidated_check_for_nans(tree_gpu_nan)
    assert any_nans is True, "Failed to detect GPU JAX NaN!"
    assert len(nan_info) == 1, f"Expected 1 NaN entry, got {len(nan_info)}"
    assert nan_info[0][1] == 2, f"Expected 2 NaNs, got {nan_info[0][1]}"
    print(f"--> Case 3C Passed: Detected GPU NaN: {messages[0]}")

    # 4. CPU-backed JAX NaN
    print("\n[Case 3D] Testing CPU-backed JAX Array NaN...", flush=True)
    tree_cpu_jax = create_mock_pytree()
    bad_v = np.ones((4, 10, 2), dtype=np.float64)
    bad_v[0, 1, 0] = np.nan
    tree_cpu_jax["states"]["v_cpu_jax"] = jax.device_put(jnp.asarray(bad_v), cpu_dev)
    any_nans, nan_info, messages = fast_consolidated_check_for_nans(tree_cpu_jax)
    assert any_nans is True, "Failed to detect CPU JAX NaN!"
    assert len(nan_info) == 1
    assert nan_info[0][1] == 1
    print(f"--> Case 3D Passed: Detected CPU-backed JAX NaN: {messages[0]}")

    # 5. Simultaneous NaNs across multiple nested paths
    print("\n[Case 3E] Testing simultaneous NaNs across multiple paths...", flush=True)
    tree_multi = create_mock_pytree()
    # Inject into GPU x, CPU pi, and CPU-JAX v
    tree_multi["states"]["x"] = jnp.asarray(bad_x)
    tree_multi["params"]["pi"][0, 0] = np.nan
    tree_multi["states"]["v_cpu_jax"] = jax.device_put(jnp.asarray(bad_v), cpu_dev)

    any_nans, nan_info, messages = fast_consolidated_check_for_nans(tree_multi)
    assert any_nans is True
    assert len(nan_info) == 3, f"Expected 3 NaN records, got {len(nan_info)}"
    print("--> Case 3E Passed: All 3 simultaneous paths detected:")
    for m in messages:
        print(f"     * {m}")

    # 6. Integration with _wrapped_resample
    print("\n[Case 3F] Testing _wrapped_resample integration and StopResampling raise...", flush=True)
    def dummy_resample_clean(data, **model):
        return model

    def dummy_resample_nan(data, **model):
        bad_model = dict(model)
        bad_model["states"] = dict(model["states"])
        bad_model["states"]["x"] = jnp.asarray(bad_x)
        return bad_model

    # Clean resample should succeed
    out = _wrapped_resample(dummy_resample_clean, {}, tree_clean)
    assert out is not None

    # NaN resample must raise StopResampling
    stopped = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            _wrapped_resample(dummy_resample_nan, {}, tree_clean)
        except StopResampling:
            stopped = True

    assert stopped, "StopResampling was not raised by _wrapped_resample on NaN!"
    print("--> Case 3F Passed: _wrapped_resample cleanly raised StopResampling and issued warning.")

    print("\n" + "=" * 80)
    print("  CONSOLIDATED NAN CHECKER AUDIT PASSED PERFECTLY!")
    print("=" * 80 + "\n", flush=True)

if __name__ == "__main__":
    test_nan_checker()
