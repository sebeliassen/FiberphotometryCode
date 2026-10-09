import os
import sys
import time
import copy
import numpy as np

# Numpy 2.x compatibility
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import jax
import jax.numpy as jnp
import jax.random as jr

import keypoint_moseq as kpms
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
from jax_moseq.models.arhmm.gibbs import resample_hdp_transitions

import utils.resample_batching as rb

PROJECT_DIR = os.path.join(KPMS_DIR, "karen_project_2")
TEST_MODEL_NAME = "test_2_sessions_model"

def sync(tensor):
    """Synchronize JAX tensor on GPU before stopping timers."""
    if hasattr(tensor, "block_until_ready"):
        tensor.block_until_ready()
    elif isinstance(tensor, (tuple, list)):
        for x in tensor:
            sync(x)
    elif isinstance(tensor, dict):
        for x in tensor.values():
            sync(x)
    return tensor

def main():
    print("=" * 80)
    print("      KPMS GIBBS SAMPLING SYNCHRONIZED TIMING & BOTTLENECK AUDIT")
    print("=" * 80)

    # 1. Load real 2-session checkpoint state in memory (READ ONLY)
    print(f"[1/4] Loading real preprocessed dataset from {TEST_MODEL_NAME}...")
    model, data, metadata, current_iter = kpms.load_checkpoint(PROJECT_DIR, TEST_MODEL_NAME)
    
    # We slice to 4 sequences (~40,000 frames) for fast, realistic benchmarking
    N_SEQS = 4
    data_slice = {
        "Y": jnp.array(data["Y"][:N_SEQS]),
        "mask": jnp.array(data["mask"][:N_SEQS]),
        "conf": jnp.array(data["conf"][:N_SEQS])
    }
    states_slice = {
        "x": jnp.array(model["states"]["x"][:N_SEQS]),
        "z": jnp.array(model["states"]["z"][:N_SEQS]),
        "v": jnp.array(model["states"]["v"][:N_SEQS]),
        "h": jnp.array(model["states"]["h"][:N_SEQS]),
        "s": jnp.array(model["states"]["s"][:N_SEQS]),
    }
    params = {k: jnp.array(v) for k, v in model["params"].items()}
    hypparams = model["hypparams"]
    raw_prior = model.get("noise_prior", 1.0)
    noise_prior = jnp.array(raw_prior[:N_SEQS]) if hasattr(raw_prior, "shape") else raw_prior
    
    total_frames = int(data_slice["mask"].sum())
    print(f"      Benchmark dataset: {N_SEQS} sequences, {total_frames:,} active frames.")
    
    seed = jr.PRNGKey(101)
    
    # Check original functions before any patching
    orig_x_fn = slds_gibbs.resample_continuous_stateseqs
    orig_z_fn = arhmm_gibbs.resample_discrete_stateseqs
    orig_ar_fn = arhmm_gibbs.resample_ar_params
    orig_v_fn = slds_gibbs.resample_location
    orig_h_fn = slds_gibbs.resample_heading
    orig_s_fn = slds_gibbs.resample_scales

    patched_x_fn = rb.pure_numba_resample_continuous_stateseqs
    patched_z_fn = rb.fast_resample_discrete_stateseqs
    patched_ar_fn = rb.vectorized_resample_ar_params
    patched_v_fn = rb.batched_gpu_resample_location
    patched_h_fn = rb.batched_resample_heading
    patched_s_fn = rb.batched_resample_scales

    results = []

    def benchmark_step(name, func, args, kwargs, n_runs=3):
        # Warmup (triggers JIT compilation or Numba compilation)
        s_warm = jr.fold_in(seed, 0)
        out = func(s_warm, *args, **kwargs)
        sync(out)
        
        # Timed runs
        durations = []
        for r in range(n_runs):
            s_run = jr.fold_in(seed, r + 1)
            t0 = time.perf_counter()
            out = func(s_run, *args, **kwargs)
            sync(out)
            t1 = time.perf_counter()
            durations.append(t1 - t0)
        return float(np.mean(durations)), out

    print("\n[2/4] Benchmarking each Gibbs sampling sub-step (Warmup + 3 synchronized runs)...")

    # Step 1: Transitions (pi, betas) - Shared
    print("  * Transitions (pi, betas)...")
    t_trans, _ = benchmark_step(
        "Transitions (pi, betas)",
        resample_hdp_transitions,
        [],
        dict(**data_slice, **states_slice, **params, **hypparams["trans_hypparams"])
    )

    # Step 2: AR Dynamics (Ab, Q)
    print("  * AR Dynamics (Ab, Q)...")
    ar_kwargs = dict(
        mask=data_slice["mask"],
        x=states_slice["x"],
        z=states_slice["z"],
        **hypparams["ar_hypparams"]
    )
    t_ar_orig, _ = benchmark_step(
        "AR Dynamics (Original JAX)",
        orig_ar_fn,
        [],
        ar_kwargs
    )
    t_ar_patch, _ = benchmark_step(
        "AR Dynamics (Numba Vectorized)",
        patched_ar_fn,
        [],
        ar_kwargs
    )
    # Estimate temporary array allocations for AR
    nlags = hypparams["ar_hypparams"].get("nlags", 3)
    num_states = params["Ab"].shape[0]
    D = states_slice["x"].shape[-1]
    N, T = states_slice["x"].shape[:2]
    ar_temp_bytes = (N * (T - nlags) * (nlags * D + 1) * 8) + (N * (T - nlags) * D * 8)

    # Step 3: Discrete States (z)
    print("  * Discrete States (z)...")
    z_args = [states_slice["x"], data_slice["mask"], params["Ab"], params["Q"], params["pi"]]
    z_kwargs = dict(nlags=nlags, num_states=num_states)
    t_z_orig, _ = benchmark_step(
        "Discrete States (Original)",
        orig_z_fn,
        z_args,
        z_kwargs
    )
    t_z_patch, _ = benchmark_step(
        "Discrete States (Tensor-Core)",
        patched_z_fn,
        z_args,
        z_kwargs
    )

    # Step 4: Continuous Latent States (x)
    print("  * Continuous Latents (x - Kalman vs Banded Cholesky)...")
    x_args = [
        data_slice["Y"], data_slice["mask"], states_slice["v"], states_slice["h"],
        states_slice["s"], states_slice["z"], params["Cd"], params["sigmasq"],
        params["Ab"], params["Q"]
    ]
    t_x_orig, _ = benchmark_step(
        "Continuous States (Original JAX Kalman)",
        orig_x_fn,
        x_args,
        dict(jitter=1e-3, parallel_message_passing=False)
    )
    t_x_patch, _ = benchmark_step(
        "Continuous States (Numba Banded Cholesky)",
        patched_x_fn,
        x_args,
        dict(jitter=1e-3)
    )

    # Step 5: Pose Location (v)
    print("  * Location (v)...")
    v_args = [
        data_slice["Y"], data_slice["mask"], states_slice["x"], states_slice["h"],
        states_slice["s"], params["Cd"], params["sigmasq"]
    ]
    v_kwargs = dict(sigmasq_loc=hypparams["cen_hypparams"]["sigmasq_loc"])
    t_v_orig, _ = benchmark_step(
        "Location (Original)",
        orig_v_fn,
        v_args,
        v_kwargs
    )
    t_v_patch, _ = benchmark_step(
        "Location (Batched GPU)",
        patched_v_fn,
        v_args,
        v_kwargs
    )

    # Step 6: Heading (h)
    print("  * Heading (h)...")
    h_args = [
        data_slice["Y"], states_slice["x"], states_slice["v"], states_slice["s"],
        params["Cd"], params["sigmasq"]
    ]
    t_h_orig, _ = benchmark_step(
        "Heading (Original)",
        orig_h_fn,
        h_args,
        dict()
    )
    t_h_patch, _ = benchmark_step(
        "Heading (Batched GPU)",
        patched_h_fn,
        h_args,
        dict()
    )

    # Step 7: Noise Scales (s)
    print("  * Noise Scales (s)...")
    s_args = [
        data_slice["Y"], states_slice["x"], states_slice["v"], states_slice["h"],
        params["Cd"], params["sigmasq"], hypparams["obs_hypparams"]["nu_s"], noise_prior
    ]
    t_s_orig, _ = benchmark_step(
        "Noise Scales (Original)",
        orig_s_fn,
        s_args,
        dict()
    )
    t_s_patch, _ = benchmark_step(
        "Noise Scales (Batched GPU)",
        patched_s_fn,
        s_args,
        dict()
    )

    # [3/4] Checkpoint & Plotting Overhead Analysis
    print("\n[3/4] Measuring Checkpoint I/O & Plotting Overhead...")
    import tempfile
    from keypoint_moseq.viz import plot_progress
    from keypoint_moseq.io import save_hdf5

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_ckpt = os.path.join(tmpdir, "test_ckpt.h5")
        
        # Time saving checkpoint
        t0 = time.perf_counter()
        save_hdf5(fake_ckpt, model, "model_snapshots/1", exist_ok=True)
        t_save = time.perf_counter() - t0
        
        # Time plotting progress (which default kpms runs when iteration == num_iters)
        t0 = time.perf_counter()
        try:
            plot_progress(
                model, data_slice, fake_ckpt, 1,
                tmpdir, "test_model", savefig=True
            )
            t_plot = time.perf_counter() - t0
        except Exception as e:
            t_plot = 0.0
            print(f"      (plot_progress test skipped/failed: {e})")

    # Compute Totals
    total_orig = t_trans + t_ar_orig + t_z_orig + t_x_orig + t_v_orig + t_h_orig + t_s_orig
    total_patch = t_trans + t_ar_patch + t_z_patch + t_x_patch + t_v_patch + t_h_patch + t_s_patch

    # [4/4] Output Report Table
    print("\n" + "=" * 90)
    print(f"{'SYNCHRONIZED GIBBS SUB-STEP TIMING BREAKDOWN (N=4 seqs, ~40,000 frames)':^90}")
    print("=" * 90)
    print(f"{'Sub-Step / Component':<34} | {'Original (s)':<14} | {'Patched (s)':<14} | {'Speedup':<10}")
    print("-" * 90)
    print(f"{'Continuous Latents (x - Kalman)':<34} | {t_x_orig:12.4f} s | {t_x_patch:12.4f} s | {t_x_orig/t_x_patch:8.2f}x")
    print(f"{'Discrete States (z - HMM)':<34} | {t_z_orig:12.4f} s | {t_z_patch:12.4f} s | {t_z_orig/t_z_patch:8.2f}x")
    print(f"{'AR Dynamics (Ab, Q)':<34} | {t_ar_orig:12.4f} s | {t_ar_patch:12.4f} s | {t_ar_orig/t_ar_patch:8.2f}x")
    print(f"{'Pose Location (v)':<34} | {t_v_orig:12.4f} s | {t_v_patch:12.4f} s | {t_v_orig/t_v_patch:8.2f}x")
    print(f"{'Pose Heading (h)':<34} | {t_h_orig:12.4f} s | {t_h_patch:12.4f} s | {t_h_orig/t_h_patch:8.2f}x")
    print(f"{'Noise Scales (s)':<34} | {t_s_orig:12.4f} s | {t_s_patch:12.4f} s | {t_s_orig/t_s_patch:8.2f}x")
    print(f"{'Transitions (pi, betas)':<34} | {t_trans:12.4f} s | {t_trans:12.4f} s | {'1.00x':>8}")
    print("-" * 90)
    print(f"{'TOTAL CORE COMPUTE ITERATION':<34} | {total_orig:12.4f} s | {total_patch:12.4f} s | {total_orig/total_patch:8.2f}x")
    print("=" * 90)
    print(f"\nEXTRA OVERHEAD PER END-OF-CHUNK ITERATION:")
    print(f"  * Checkpoint Save (save_hdf5):       {t_save:6.3f} s")
    print(f"  * Progress Plots (plot_progress):     {t_plot:6.3f} s")
    print(f"  * AR Temporary Array Footprint:      {ar_temp_bytes/1e6:6.2f} MB (per update for N={N_SEQS})")
    print("=" * 90)

if __name__ == "__main__":
    main()
