#!/usr/bin/env python3
"""
diagnostics/test_cpu_cache_validity.py

Correctness test for CPU-Cache validity across:
1. Successive AR-only iterations (x is fixed; verify cache persistence and exact AR statistics).
2. After an SLDS update changes x (verify cache invalidation, new x retrieval, and AR stat parity).
3. After an FFBS update changes z (verify new z retrieval, transition counts, and AR stat parity).
4. After loading a checkpoint / switching models (verify cache clean-switch and no state leakage).
"""

import sys, os
sys.path.insert(0, '/home/fsp585/Desktop/FiberphotometryCode')
sys.path.insert(0, '/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll')

import numpy as np
if not hasattr(np, "bool8"): np.bool8 = np.bool_

import jax
import jax.numpy as jnp
import jax.random as jr
import keypoint_moseq as kpms
import utils.resample_batching as rb
from utils.resample_batching import KPMSStateCache

PROJECT_DIR = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project"
SOURCE_MODEL = "amph_benchmark_test_timing"
if not os.path.exists(os.path.join(PROJECT_DIR, SOURCE_MODEL)):
    SOURCE_MODEL = "amph_prod_k150k_run_01"

def test_cache_validity():
    print("=" * 80)
    print("  TEST 1: CPU-CACHE VALIDITY & GIBBS STATE LIFECYCLE AUDIT")
    print("=" * 80, flush=True)

    rb.apply_optimizations()
    KPMSStateCache.clear()

    print("Loading test slice from production checkpoint...", flush=True)
    model, data, metadata, _ = kpms.load_checkpoint(PROJECT_DIR, SOURCE_MODEL)

    # Use a 4-sequence slice with 500 frames for fast, deterministic testing
    N_test, T_test = 4, 500
    slice_data = {
        "Y": np.asarray(data["Y"][:N_test, :T_test]),
        "mask": np.asarray(data["mask"][:N_test, :T_test], dtype=bool),
        "conf": np.asarray(data["conf"][:N_test, :T_test])
    }
    slice_model = {
        "seed": model["seed"],
        "states": {
            "x": jnp.asarray(model["states"]["x"][:N_test, :T_test]),
            "z": jnp.asarray(model["states"]["z"][:N_test, :T_test - 3]),
            "v": jnp.asarray(model["states"]["v"][:N_test, :T_test]),
            "h": jnp.asarray(model["states"]["h"][:N_test, :T_test]),
            "s": jnp.asarray(model["states"]["s"][:N_test, :T_test]),
        },
        "params": {
            "Ab": jnp.asarray(model["params"]["Ab"]),
            "Q": jnp.asarray(model["params"]["Q"]),
            "pi": jnp.asarray(model["params"]["pi"]),
            "betas": jnp.asarray(model["params"]["betas"]),
            "Cd": jnp.asarray(model["params"]["Cd"]),
            "sigmasq": jnp.asarray(model["params"]["sigmasq"]),
        },
        "hypparams": model["hypparams"],
        "noise_prior": model["noise_prior"],
    }

    # -------------------------------------------------------------------------
    # Test 1A: Initial Cache Population and AR Statistics Parity
    # -------------------------------------------------------------------------
    print("\n[Case 1A] Testing initial cache population & AR stats parity...", flush=True)
    KPMSStateCache.clear()

    # Fresh computation without cache
    fresh_x = np.asarray(slice_model["states"]["x"])
    fresh_z = np.asarray(slice_model["states"]["z"])
    fresh_mask = np.asarray(slice_data["mask"])

    fresh_counts, fresh_S_ii, fresh_S_oi, fresh_S_oo = rb._zero_alloc_accumulate_ar_stats(
        fresh_x, fresh_z, fresh_mask, nlags=3, num_states=100
    )

    # Cached access
    cached_x = KPMSStateCache.get_x(slice_model["states"]["x"])
    cached_z = KPMSStateCache.get_z(slice_model["states"]["z"])
    cached_mask = KPMSStateCache.get_mask(slice_data["mask"])

    assert np.array_equal(cached_x, fresh_x), "Cached x does not match fresh x!"
    assert np.array_equal(cached_z, fresh_z), "Cached z does not match fresh z!"
    assert np.array_equal(cached_mask, fresh_mask), "Cached mask does not match fresh mask!"

    cached_counts, cached_S_ii, cached_S_oi, cached_S_oo = rb._zero_alloc_accumulate_ar_stats(
        cached_x, cached_z, cached_mask, nlags=3, num_states=100
    )

    np.testing.assert_array_equal(cached_counts, fresh_counts)
    np.testing.assert_allclose(cached_S_ii, fresh_S_ii, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(cached_S_oi, fresh_S_oi, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(cached_S_oo, fresh_S_oo, rtol=1e-14, atol=1e-14)
    print("--> Case 1A Passed: Cached inputs and AR statistics match fresh values exactly.")

    # -------------------------------------------------------------------------
    # Test 1B: Successive AR-only iterations (x fixed, z updated)
    # -------------------------------------------------------------------------
    print("\n[Case 1B] Testing successive AR-only iterations...", flush=True)
    # In AR-only, x is fixed. We simulate an FFBS update that changes z.
    old_x_id = id(slice_model["states"]["x"])
    old_z_cpu = np.copy(KPMSStateCache.get_z(slice_model["states"]["z"]))

    # Resample z using fast_resample_discrete_stateseqs
    new_z_dev = rb.fast_resample_discrete_stateseqs(
        slice_model["seed"], slice_model["states"]["x"], slice_data["mask"],
        slice_model["params"]["Ab"], slice_model["params"]["Q"], slice_model["params"]["pi"],
        nlags=3, num_states=100, batch_size=2
    )
    slice_model["states"]["z"] = new_z_dev

    # Verify x cache was preserved without re-conversion
    retained_x = KPMSStateCache.get_x(slice_model["states"]["x"])
    assert retained_x is cached_x, "x cache was needlessly reallocated in AR-only mode!"

    # Verify z cache was updated to the new z
    new_z_cached = KPMSStateCache.get_z(slice_model["states"]["z"])
    new_z_fresh = np.asarray(new_z_dev)
    assert np.array_equal(new_z_cached, new_z_fresh), "z cache did not match freshly downloaded z!"
    assert not np.array_equal(new_z_cached, old_z_cpu), "z cache was not updated after FFBS!"

    # Verify AR stats match with new z
    counts_ar2, S_ii_ar2, _, _ = rb._zero_alloc_accumulate_ar_stats(
        retained_x, new_z_cached, cached_mask, nlags=3, num_states=100
    )
    counts_fresh2, S_ii_fresh2, _, _ = rb._zero_alloc_accumulate_ar_stats(
        fresh_x, new_z_fresh, fresh_mask, nlags=3, num_states=100
    )
    np.testing.assert_array_equal(counts_ar2, counts_fresh2)
    np.testing.assert_allclose(S_ii_ar2, S_ii_fresh2, rtol=1e-14, atol=1e-14)
    print("--> Case 1B Passed: Successive AR iteration preserves fixed x and accurately refreshes z.")

    # -------------------------------------------------------------------------
    # Test 1C: After an SLDS update changes x
    # -------------------------------------------------------------------------
    print("\n[Case 1C] Testing SLDS update changing x...", flush=True)
    old_x_cpu = np.copy(KPMSStateCache.get_x(slice_model["states"]["x"]))

    # Resample continuous stateseqs
    new_x_dev = rb.pure_numba_resample_continuous_stateseqs(
        slice_model["seed"], slice_data["Y"], slice_data["mask"],
        slice_model["states"]["v"], slice_model["states"]["h"], slice_model["states"]["s"],
        slice_model["states"]["z"], slice_model["params"]["Cd"], slice_model["params"]["sigmasq"],
        slice_model["params"]["Ab"], slice_model["params"]["Q"], jitter=1e-3, chunk_size=2
    )
    slice_model["states"]["x"] = new_x_dev

    # Verify that cache now contains the NEW x, not the old x
    new_x_cached = KPMSStateCache.get_x(slice_model["states"]["x"])
    new_x_fresh = np.asarray(new_x_dev)

    assert np.array_equal(new_x_cached, new_x_fresh), "x cache does not match new x from SLDS!"
    diff_from_old = np.max(np.abs(new_x_cached - old_x_cpu))
    assert diff_from_old > 1e-4, f"x did not change after SLDS resampling! diff={diff_from_old}"

    # Verify AR statistics with new x
    counts_slds, S_ii_slds, _, _ = rb._zero_alloc_accumulate_ar_stats(
        new_x_cached, new_z_cached, cached_mask, nlags=3, num_states=100
    )
    counts_fresh_slds, S_ii_fresh_slds, _, _ = rb._zero_alloc_accumulate_ar_stats(
        new_x_fresh, new_z_fresh, fresh_mask, nlags=3, num_states=100
    )
    np.testing.assert_array_equal(counts_slds, counts_fresh_slds)
    np.testing.assert_allclose(S_ii_slds, S_ii_fresh_slds, rtol=1e-14, atol=1e-14)
    print("--> Case 1C Passed: SLDS update cleanly replaces x cache and produces exact AR statistics.")

    # -------------------------------------------------------------------------
    # Test 1D: Loading a checkpoint / Switching models
    # -------------------------------------------------------------------------
    print("\n[Case 1D] Testing checkpoint load / model switching...", flush=True)
    # Simulate loading Model B with different randomized states
    rng = np.random.default_rng(999)
    model_B_x = jnp.asarray(rng.standard_normal((N_test, T_test, 10)))
    model_B_z = jnp.asarray(rng.integers(0, 100, size=(N_test, T_test - 3), dtype=np.int32))

    # KPMSStateCache should NOT return old model's x or z
    # When a new DeviceArray with a different ID is queried:
    cached_B_x = KPMSStateCache.get_x(model_B_x)
    cached_B_z = KPMSStateCache.get_z(model_B_z)

    assert np.array_equal(cached_B_x, np.asarray(model_B_x)), "Cache served stale x after model switch!"
    assert np.array_equal(cached_B_z, np.asarray(model_B_z)), "Cache served stale z after model switch!"
    assert not np.array_equal(cached_B_x, new_x_cached), "Cache leaked previous model's x!"
    assert not np.array_equal(cached_B_z, new_z_cached), "Cache leaked previous model's z!"

    # Explicit clear test
    KPMSStateCache.clear()
    assert KPMSStateCache._cached_x is None, "Cache clear failed for x!"
    assert KPMSStateCache._cached_z is None, "Cache clear failed for z!"
    print("--> Case 1D Passed: Model switching and checkpoint reload cleanly invalidate old states.")

    print("\n" + "=" * 80)
    print("  ALL CPU-CACHE VALIDITY AUDITS PASSED PERFECTLY!")
    print("=" * 80 + "\n", flush=True)

if __name__ == "__main__":
    test_cache_validity()
