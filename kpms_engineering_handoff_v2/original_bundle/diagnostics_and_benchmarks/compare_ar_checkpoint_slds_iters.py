import os
import sys
import copy
import time
import numpy as np
np.bool8 = np.bool_
import jax
import jax.numpy as jnp
import jax.random as jr
import h5py

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

def calculate_duration_stats(z, mask):
    durations = []
    for i in range(len(z)):
        seq_z = z[i]
        seq_m = mask[i] if mask.ndim > 1 else np.ones_like(seq_z, dtype=bool)
        valid_idx = np.where(seq_m)[0]
        if len(valid_idx) == 0: continue
        
        curr_s = seq_z[valid_idx[0]]
        curr_len = 1
        for j in range(1, len(valid_idx)):
            if valid_idx[j] == valid_idx[j-1] + 1 and seq_z[valid_idx[j]] == curr_s:
                curr_len += 1
            else:
                durations.append(curr_len)
                curr_s = seq_z[valid_idx[j]]
                curr_len = 1
        durations.append(curr_len)
    durations = np.array(durations)
    return {
        "median_frames": float(np.median(durations)),
        "mean_frames": float(np.mean(durations)),
        "median_ms": float(np.median(durations) * 25.0), # 40 Hz -> 25ms/frame
        "mean_ms": float(np.mean(durations) * 25.0),
        "total_bouts": len(durations)
    }

def run_comparative_experiment():
    print("=" * 85)
    print("  LOADING AR CHECKPOINT (Iteration 50) FROM final_karen_model_k150k")
    print("=" * 85)
    
    # 1. Load full checkpoint at iteration 50
    model, data, metadata, current_iter = kpms.load_checkpoint(PROJECT_DIR, MODEL_NAME, iteration=50)
    
    # 2. Slice to 30 videos (~300k frames) for safe unpatched GPU Kalman memory
    N_SLICED = 30
    print(f"Slicing dataset to first {N_SLICED} videos for exact GPU Kalman memory budgeting...")
    sliced_data = {
        'Y': data['Y'][:N_SLICED],
        'mask': data['mask'][:N_SLICED]
    }
    if 'conf' in data and data['conf'] is not None:
        sliced_data['conf'] = data['conf'][:N_SLICED]
        
    sliced_metadata = tuple(m[:N_SLICED] for m in metadata) if isinstance(metadata, tuple) else metadata
    
    # Slice model states and noise_prior
    sliced_states = {}
    for k, v in model['states'].items():
        if hasattr(v, '__len__') and len(v) >= N_SLICED:
            sliced_states[k] = v[:N_SLICED]
        else:
            sliced_states[k] = v
            
    if 'noise_prior' in model and hasattr(model['noise_prior'], '__len__') and len(model['noise_prior']) >= N_SLICED:
        model['noise_prior'] = model['noise_prior'][:N_SLICED]
            
    # Compute initial baseline stats at AR step 50
    nlags = model['hypparams']['ar_hypparams']['nlags']
    mask_z = sliced_data['mask'][:, nlags:]
    z_init = np.array(sliced_states['z'])
    init_stats = calculate_duration_stats(z_init, mask_z)
    
    unq_z, cnts = np.unique(z_init[mask_z > 0], return_counts=True)
    active_99 = np.sum(np.cumsum(np.sort(cnts)[::-1]) / len(z_init[mask_z > 0]) < 0.99) + 1
    
    print(f"\n--- Initial AR Checkpoint State (Iteration 50) ---")
    print(f"  * Median Duration: {init_stats['median_frames']:.1f} frames ({init_stats['median_ms']:.1f} ms @ 40Hz)")
    print(f"  * Mean Duration:   {init_stats['mean_frames']:.2f} frames ({init_stats['mean_ms']:.1f} ms @ 40Hz)")
    print(f"  * Active Syllables (99% usage): {active_99} (Total unique: {len(unq_z)})")
    print(f"  * Total Bouts:     {init_stats['total_bouts']:,}")
    print("=" * 85)
    
    # Save original references
    orig_resample_continuous = slds_gibbs.resample_continuous_stateseqs
    orig_resample_discrete = arhmm_gibbs.resample_discrete_stateseqs
    orig_resample_ar = arhmm_gibbs.resample_ar_params
    orig_resample_scales = slds_gibbs.resample_scales
    orig_resample_heading = slds_gibbs.resample_heading
    orig_resample_loc = slds_gibbs.resample_location

    NUM_ITERS = 2
    
    # =========================================================================
    # BRANCH A: UNPATCHED ORIGINAL JAX KEYPOINT-MOSEQ
    # =========================================================================
    print("\n" + "=" * 85)
    print("  BRANCH A: UNPATCHED ORIGINAL JAX KPMS (2 SLDS Iterations)")
    print("=" * 85)
    
    # Ensure unpatched references are active
    slds_gibbs.resample_continuous_stateseqs = orig_resample_continuous
    arhmm_gibbs.resample_discrete_stateseqs = orig_resample_discrete
    arhmm_gibbs.resample_ar_params = orig_resample_ar
    slds_gibbs.resample_scales = orig_resample_scales
    slds_gibbs.resample_heading = orig_resample_heading
    slds_gibbs.resample_location = orig_resample_loc
    
    model_unpatched = copy.deepcopy(model)
    model_unpatched['states'] = copy.deepcopy(sliced_states)
    
    out_dir_unpatched = os.path.join(KPMS_DIR, "scratch_unpatched_test")
    os.makedirs(out_dir_unpatched, exist_ok=True)
    
    t0_unpatched = time.perf_counter()
    model_unpatched, _ = kpms.fit_model(
        model_unpatched, sliced_data, sliced_metadata, out_dir_unpatched,
        model_name="unpatched_test",
        ar_only=False,
        start_iter=50,
        num_iters=50 + NUM_ITERS,
        parallel_message_passing=False,
        save_every_n_iters=10
    )
    time_unpatched = time.perf_counter() - t0_unpatched
    
    z_unp = np.array(model_unpatched['states']['z'])
    stats_unp = calculate_duration_stats(z_unp, mask_z)
    unq_unp, cnts_unp = np.unique(z_unp[mask_z > 0], return_counts=True)
    active_99_unp = np.sum(np.cumsum(np.sort(cnts_unp)[::-1]) / len(z_unp[mask_z > 0]) < 0.99) + 1
    
    print(f"\n[Unpatched Complete in {time_unpatched:.2f}s]")
    print(f"  * Median Duration: {stats_unp['median_frames']:.1f} frames ({stats_unp['median_ms']:.1f} ms)")
    print(f"  * Mean Duration:   {stats_unp['mean_frames']:.2f} frames ({stats_unp['mean_ms']:.1f} ms)")
    print(f"  * Active Syllables (99%): {active_99_unp}")
    print(f"  * Total Bouts:     {stats_unp['total_bouts']:,}")
    
    # =========================================================================
    # BRANCH B: PATCHED & FIXED (ASCENDING LAG ORDER)
    # =========================================================================
    print("\n" + "=" * 85)
    print("  BRANCH B: PATCHED & FIXED OPTIMIZED KPMS (2 SLDS Iterations)")
    print("=" * 85)
    
    # Define fixed ascending get_lags
    def _numpy_get_lags_fixed(x_np, nlags):
        N, T, D = x_np.shape
        lags = [x_np[:, i : T - nlags + i, :] for i in range(nlags)]
        return np.concatenate(lags, axis=-1)
    rb._numpy_get_lags = _numpy_get_lags_fixed
    apply_optimizations()
    
    model_patched_fixed = copy.deepcopy(model)
    model_patched_fixed['states'] = copy.deepcopy(sliced_states)
    
    out_dir_fixed = os.path.join(KPMS_DIR, "scratch_patched_fixed_test")
    os.makedirs(out_dir_fixed, exist_ok=True)
    
    t0_fixed = time.perf_counter()
    model_patched_fixed, _ = kpms.fit_model(
        model_patched_fixed, sliced_data, sliced_metadata, out_dir_fixed,
        model_name="patched_fixed_test",
        ar_only=False,
        start_iter=50,
        num_iters=50 + NUM_ITERS,
        parallel_message_passing=False,
        save_every_n_iters=10
    )
    time_fixed = time.perf_counter() - t0_fixed
    
    z_fix = np.array(model_patched_fixed['states']['z'])
    stats_fix = calculate_duration_stats(z_fix, mask_z)
    unq_fix, cnts_fix = np.unique(z_fix[mask_z > 0], return_counts=True)
    active_99_fix = np.sum(np.cumsum(np.sort(cnts_fix)[::-1]) / len(z_fix[mask_z > 0]) < 0.99) + 1
    
    print(f"\n[Patched (Fixed) Complete in {time_fixed:.2f}s]")
    print(f"  * Median Duration: {stats_fix['median_frames']:.1f} frames ({stats_fix['median_ms']:.1f} ms)")
    print(f"  * Mean Duration:   {stats_fix['mean_frames']:.2f} frames ({stats_fix['mean_ms']:.1f} ms)")
    print(f"  * Active Syllables (99%): {active_99_fix}")
    print(f"  * Total Bouts:     {stats_fix['total_bouts']:,}")
    
    # =========================================================================
    # BRANCH C: BUGGY UNFIXED PATCH (REVERSED LAG ORDER) - NEGATIVE CONTROL
    # =========================================================================
    print("\n" + "=" * 85)
    print("  BRANCH C: BUGGY UNFIXED PATCH (REVERSED LAGS) - NEGATIVE CONTROL")
    print("=" * 85)
    
    # Re-inject the buggy reversed get_lags
    def _numpy_get_lags_buggy(x_np, nlags):
        N, T, D = x_np.shape
        lags = [x_np[:, nlags - 1 - i : T - 1 - i, :] for i in range(nlags)]
        return np.concatenate(lags, axis=-1)
    rb._numpy_get_lags = _numpy_get_lags_buggy
    apply_optimizations()
    
    model_patched_buggy = copy.deepcopy(model)
    model_patched_buggy['states'] = copy.deepcopy(sliced_states)
    
    out_dir_buggy = os.path.join(KPMS_DIR, "scratch_patched_buggy_test")
    os.makedirs(out_dir_buggy, exist_ok=True)
    
    t0_buggy = time.perf_counter()
    model_patched_buggy, _ = kpms.fit_model(
        model_patched_buggy, sliced_data, sliced_metadata, out_dir_buggy,
        model_name="patched_buggy_test",
        ar_only=False,
        start_iter=50,
        num_iters=50 + NUM_ITERS,
        parallel_message_passing=False,
        save_every_n_iters=10
    )
    time_buggy = time.perf_counter() - t0_buggy
    
    z_bug = np.array(model_patched_buggy['states']['z'])
    stats_bug = calculate_duration_stats(z_bug, mask_z)
    unq_bug, cnts_bug = np.unique(z_bug[mask_z > 0], return_counts=True)
    active_99_bug = np.sum(np.cumsum(np.sort(cnts_bug)[::-1]) / len(z_bug[mask_z > 0]) < 0.99) + 1
    
    print(f"\n[Buggy Patch Complete in {time_buggy:.2f}s]")
    print(f"  * Median Duration: {stats_bug['median_frames']:.1f} frames ({stats_bug['median_ms']:.1f} ms)")
    print(f"  * Mean Duration:   {stats_bug['mean_frames']:.2f} frames ({stats_bug['mean_ms']:.1f} ms)")
    print(f"  * Active Syllables (99%): {active_99_bug}")
    print(f"  * Total Bouts:     {stats_bug['total_bouts']:,}")

    # Restore fixed get_lags globally
    rb._numpy_get_lags = _numpy_get_lags_fixed
    apply_optimizations()

    # =========================================================================
    # SUMMARY COMPARISON TABLE
    # =========================================================================
    print("\n" + "=" * 105)
    print("                      HEAD-TO-HEAD COMPARISON (2 SLDS ITERATIONS FROM ITER 50)")
    print("=" * 105)
    print(f"{'Metric':<25} | {'Initial (Iter 50)':<18} | {'Unpatched JAX':<18} | {'Patched (FIXED)':<18} | {'Buggy Patch (OLD)':<18}")
    print("-" * 105)
    print(f"{'Median Duration':<25} | {init_stats['median_frames']:4.1f}f ({init_stats['median_ms']:5.1f}ms)  | {stats_unp['median_frames']:4.1f}f ({stats_unp['median_ms']:5.1f}ms)  | {stats_fix['median_frames']:4.1f}f ({stats_fix['median_ms']:5.1f}ms)  | {stats_bug['median_frames']:4.1f}f ({stats_bug['median_ms']:5.1f}ms)")
    print(f"{'Mean Duration':<25} | {init_stats['mean_frames']:4.2f}f ({init_stats['mean_ms']:5.1f}ms)  | {stats_unp['mean_frames']:4.2f}f ({stats_unp['mean_ms']:5.1f}ms)  | {stats_fix['mean_frames']:4.2f}f ({stats_fix['mean_ms']:5.1f}ms)  | {stats_bug['mean_frames']:4.2f}f ({stats_bug['mean_ms']:5.1f}ms)")
    print(f"{'Active Syllables (99%)':<25} | {active_99:<18} | {active_99_unp:<18} | {active_99_fix:<18} | {active_99_bug:<18}")
    print(f"{'Total Bouts':<25} | {init_stats['total_bouts']:<18,d} | {stats_unp['total_bouts']:<18,d} | {stats_fix['total_bouts']:<18,d} | {stats_bug['total_bouts']:<18,d}")
    print(f"{'Runtime (2 SLDS Iters)':<25} | {'N/A':<18} | {time_unpatched:15.2f}s | {time_fixed:15.2f}s | {time_buggy:15.2f}s")
    print(f"{'Speedup vs JAX':<25} | {'N/A':<18} | {'1.00x':<18} | {time_unpatched/time_fixed:17.2f}x | {time_unpatched/time_buggy:17.2f}x")
    print("=" * 105)

if __name__ == "__main__":
    run_comparative_experiment()
