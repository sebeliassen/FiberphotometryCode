import os
import sys
import copy
import time
import numpy as np
np.bool8 = np.bool_
import jax
import jax.numpy as jnp
import jax.random as jr

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import keypoint_moseq as kpms
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
from utils.resample_batching import apply_optimizations
import utils.resample_batching as rb

PROJECT_DIR = os.path.join(KPMS_DIR, "karen_project_2")
MODEL_NAME = "final_karen_model_k150k"

def benchmark_iterations():
    print("=" * 80)
    print("  KPMS STEADY-STATE ITERATION TIMING BENCHMARK (5 SLDS ITERATIONS)")
    print("=" * 80)
    
    # 1. Load checkpoint at Iteration 50
    model, data, metadata, _ = kpms.load_checkpoint(PROJECT_DIR, MODEL_NAME, iteration=50)
    
    N_SLICED = 30
    sliced_data = {
        'Y': data['Y'][:N_SLICED],
        'mask': data['mask'][:N_SLICED]
    }
    if 'conf' in data and data['conf'] is not None:
        sliced_data['conf'] = data['conf'][:N_SLICED]
        
    sliced_metadata = tuple(m[:N_SLICED] for m in metadata) if isinstance(metadata, tuple) else metadata
    
    sliced_states = {}
    for k, v in model['states'].items():
        if hasattr(v, '__len__') and len(v) >= N_SLICED:
            sliced_states[k] = v[:N_SLICED]
        else:
            sliced_states[k] = v
            
    if 'noise_prior' in model and hasattr(model['noise_prior'], '__len__') and len(model['noise_prior']) >= N_SLICED:
        model['noise_prior'] = model['noise_prior'][:N_SLICED]

    # Save original functions
    orig_resample_continuous = slds_gibbs.resample_continuous_stateseqs
    orig_resample_discrete = arhmm_gibbs.resample_discrete_stateseqs
    orig_resample_ar = arhmm_gibbs.resample_ar_params
    orig_resample_scales = slds_gibbs.resample_scales
    orig_resample_heading = slds_gibbs.resample_heading
    orig_resample_loc = slds_gibbs.resample_location

    NUM_ITERS = 4

    # -------------------------------------------------------------------------
    # TEST 1: PATCHED & FIXED (Warmup iter + Steady state)
    # -------------------------------------------------------------------------
    print("\n--- Testing Branch: PATCHED (FIXED) ---")
    apply_optimizations()
    
    m_fixed = copy.deepcopy(model)
    m_fixed['states'] = copy.deepcopy(sliced_states)
    out_dir_fixed = os.path.join(KPMS_DIR, "scratch_speed_fixed")
    os.makedirs(out_dir_fixed, exist_ok=True)
    
    fixed_times = []
    curr_m = m_fixed
    for i in range(1, NUM_ITERS + 1):
        t0 = time.perf_counter()
        curr_m, _ = kpms.fit_model(
            curr_m, sliced_data, sliced_metadata, out_dir_fixed,
            model_name="speed_fixed",
            ar_only=False,
            start_iter=50 + i - 1,
            num_iters=50 + i,
            parallel_message_passing=False,
            save_every_n_iters=100,
            verbose=False
        )
        dt = time.perf_counter() - t0
        fixed_times.append(dt)
        tag = "(JIT Compile + Run)" if i == 1 else "(Steady State)"
        print(f"  * Iteration {i}: {dt:6.3f} s  {tag}")

    # -------------------------------------------------------------------------
    # TEST 2: BUGGY OLD PATCH (Same warmup + Steady state)
    # -------------------------------------------------------------------------
    print("\n--- Testing Branch: BUGGY OLD PATCH ---")
    def _numpy_get_lags_buggy(x_np, nlags):
        N, T, D = x_np.shape
        lags = [x_np[:, nlags - 1 - i : T - 1 - i, :] for i in range(nlags)]
        return np.concatenate(lags, axis=-1)
    rb._numpy_get_lags = _numpy_get_lags_buggy
    apply_optimizations()

    m_buggy = copy.deepcopy(model)
    m_buggy['states'] = copy.deepcopy(sliced_states)
    out_dir_buggy = os.path.join(KPMS_DIR, "scratch_speed_buggy")
    os.makedirs(out_dir_buggy, exist_ok=True)

    buggy_times = []
    curr_m = m_buggy
    for i in range(1, NUM_ITERS + 1):
        t0 = time.perf_counter()
        curr_m, _ = kpms.fit_model(
            curr_m, sliced_data, sliced_metadata, out_dir_buggy,
            model_name="speed_buggy",
            ar_only=False,
            start_iter=50 + i - 1,
            num_iters=50 + i,
            parallel_message_passing=False,
            save_every_n_iters=100,
            verbose=False
        )
        dt = time.perf_counter() - t0
        buggy_times.append(dt)
        tag = "(JIT Compile + Run)" if i == 1 else "(Steady State)"
        print(f"  * Iteration {i}: {dt:6.3f} s  {tag}")

    # Restore fixed globally
    def _numpy_get_lags_fixed(x_np, nlags):
        N, T, D = x_np.shape
        lags = [x_np[:, i : T - nlags + i, :] for i in range(nlags)]
        return np.concatenate(lags, axis=-1)
    rb._numpy_get_lags = _numpy_get_lags_fixed
    apply_optimizations()

    # -------------------------------------------------------------------------
    # TEST 3: UNPATCHED ORIGINAL JAX
    # -------------------------------------------------------------------------
    print("\n--- Testing Branch: UNPATCHED ORIGINAL JAX ---")
    import jax_moseq.models.keypoint_slds as kp_slds_mod
    import jax_moseq.models.slds as slds_mod
    
    slds_gibbs.resample_continuous_stateseqs = orig_resample_continuous
    arhmm_gibbs.resample_discrete_stateseqs = orig_resample_discrete
    arhmm_gibbs.resample_ar_params = orig_resample_ar
    slds_gibbs.resample_scales = orig_resample_scales
    slds_gibbs.resample_heading = orig_resample_heading
    slds_gibbs.resample_location = orig_resample_loc
    
    kp_slds_mod.resample_continuous_stateseqs = orig_resample_continuous
    kp_slds_mod.resample_scales = orig_resample_scales
    kp_slds_mod.resample_heading = orig_resample_heading
    kp_slds_mod.resample_location = orig_resample_loc
    
    from jax_moseq.models.slds.gibbs import resample_continuous_stateseqs as slds_orig_resample_x
    slds_mod.resample_continuous_stateseqs = slds_orig_resample_x
    slds_mod.resample_scales = orig_resample_scales
    slds_mod.resample_heading = orig_resample_heading
    slds_mod.resample_location = orig_resample_loc

    m_unp = copy.deepcopy(model)
    m_unp['states'] = copy.deepcopy(sliced_states)
    out_dir_unp = os.path.join(KPMS_DIR, "scratch_speed_unp")
    os.makedirs(out_dir_unp, exist_ok=True)

    unp_times = []
    curr_m = m_unp
    for i in range(1, NUM_ITERS + 1):
        t0 = time.perf_counter()
        curr_m, _ = kpms.fit_model(
            curr_m, sliced_data, sliced_metadata, out_dir_unp,
            model_name="speed_unp",
            ar_only=False,
            start_iter=50 + i - 1,
            num_iters=50 + i,
            parallel_message_passing=False,
            save_every_n_iters=100,
            verbose=False
        )
        dt = time.perf_counter() - t0
        unp_times.append(dt)
        tag = "(JIT Compile + Run)" if i == 1 else "(Steady State)"
        print(f"  * Iteration {i}: {dt:6.3f} s  {tag}")

    # -------------------------------------------------------------------------
    # COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                 PER-ITERATION TIMING BREAKDOWN (SECONDS)")
    print("=" * 80)
    print(f"{'Iteration':<15} | {'Patched (FIXED)':<18} | {'Buggy Patch (OLD)':<18} | {'Unpatched JAX':<18}")
    print("-" * 80)
    for i in range(NUM_ITERS):
        iter_name = f"Iter {i+1} (Compile)" if i == 0 else f"Iter {i+1} (Steady)"
        print(f"{iter_name:<15} | {fixed_times[i]:16.3f}s | {buggy_times[i]:16.3f}s | {unp_times[i]:16.3f}s")
    print("-" * 80)
    
    fixed_steady = np.mean(fixed_times[1:])
    buggy_steady = np.mean(buggy_times[1:])
    unp_steady = np.mean(unp_times[1:])
    
    print(f"{'Steady-State Mean':<15} | {fixed_steady:16.3f}s | {buggy_steady:16.3f}s | {unp_steady:16.3f}s")
    print(f"{'Speedup vs JAX':<15} | {unp_steady/fixed_steady:17.2f}x | {unp_steady/buggy_steady:17.2f}x | {'1.00x':<18}")
    print("=" * 80)

if __name__ == "__main__":
    benchmark_iterations()
