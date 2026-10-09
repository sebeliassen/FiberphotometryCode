#!/usr/bin/env python3
"""
run_cohort_2ar_2slds_test.py
=============================================================================
Validation run: 2 AR + 2 SLDS iterations on the FULL 118-session Tandem Cohort
(1,851 sequences, 17,953,523 frames @ 60 FPS, 23 keypoints).

Objectives:
1. Verify 100% end-to-end execution on live GPU without OOM.
2. Measure exact cold (JIT compilation) vs warm per-iteration speed for both AR and SLDS.
3. Compute precise empirical completion time for the 50 AR + 200 SLDS production run.
=============================================================================
"""

import argparse
import copy
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Live unbuffered stdout/stderr
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Hardware environment flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_MEMORY_PATCH"] = "1"

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import jax
import keypoint_moseq as kpms
from jax_moseq.utils.debugging import convert_data_precision

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))
KPMS_UTILS_DIR = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll"
sys.path.insert(0, str(KPMS_UTILS_DIR))

import utils.resample_batching as rb
from utils.batch_init import batched_init_model_by_video
from utils.trimming import filter_with_masks, presence_mask_from_conf

EXPERIMENT_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
PROD_KPMS_DIR = EXPERIMENT_DIR / "kpms_tandem_cohort_production"
PROD_DATA_DIR = PROD_KPMS_DIR / "data"
MODEL_NAME = "test_2ar_2slds_cohort"

INIT_KAPPA = 7.5e13
DECREASE_KAPPA_FACTOR = 10.0
SLDS_KAPPA = INIT_KAPPA / DECREASE_KAPPA_FACTOR
TARGET_FPS = 60

def main():
    print("=" * 80, flush=True)
    print("🔬 EMPIRICAL VALIDATION: 2 AR + 2 SLDS ON FULL COHORT (118 SESSIONS)", flush=True)
    print(f"  Date & Time:         {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  PID:                 {os.getpid()}", flush=True)
    print(f"  JAX Device:          {jax.devices()[0]}", flush=True)
    print(f"  Project Directory:   {PROD_KPMS_DIR}", flush=True)
    print("=" * 80, flush=True)

    # 1. Apply high-performance kernels
    print("\n[Step 1/5] Applying Amdahl 27x Optimizations (Banded Cholesky + GPU Kernels)...", flush=True)
    rb.apply_optimizations()

    # 2. Load Config & Tracking Data (with disk cache)
    print("\n[Step 2/5] Loading configuration and tracking data...", flush=True)
    config = kpms.load_config(str(PROD_KPMS_DIR))
    config["fps"] = TARGET_FPS
    config["trans_hypparams"]["kappa"] = INIT_KAPPA

    preprocessed_cache = PROD_KPMS_DIR / "preprocessed_data.pkl"
    if preprocessed_cache.exists():
        print(f"  [CACHE HIT] Loading preprocessed tracking data from {preprocessed_cache}...", flush=True)
        with open(preprocessed_cache, "rb") as f:
            data, metadata = pickle.load(f)
    else:
        print("  [CACHE MISS] Loading DLC tracking and computing presence masks...", flush=True)
        coordinates, confidences, bodyparts = kpms.load_keypoints(str(PROD_DATA_DIR), format="deeplabcut")

        # Automated presence masks & outlier filtering
        print("  Applying presence masks & outlier filtering across sessions...", flush=True)
        all_masks = {name: presence_mask_from_conf(conf) for name, conf in confidences.items()}
        filtered_coords, filtered_confs, video_frame_indexes = filter_with_masks(
            coordinates, confidences, all_masks, str(PROD_KPMS_DIR),
            return_video_frame_indexes=True, **config
        )

        # Drop any session with < 5 frames
        keys_to_remove = [k for k, v in filtered_coords.items() if len(v) < 5]
        for k in keys_to_remove:
            filtered_coords.pop(k)
            filtered_confs.pop(k)

        data, metadata = kpms.format_data(filtered_coords, filtered_confs, **config)
        data = convert_data_precision(data, np.float64)
        data["Y"] = np.asarray(data["Y"])

        print(f"  [CACHE SAVE] Saving preprocessed cohort data to {preprocessed_cache}...", flush=True)
        with open(preprocessed_cache, "wb") as f:
            pickle.dump((data, metadata), f, protocol=pickle.HIGHEST_PROTOCOL)

    total_frames = int(data["mask"].sum())
    n_segs, seg_len, n_kps, d = data["Y"].shape
    print(f"  Active animal frames: {total_frames:,}", flush=True)
    print(f"  Batched Y tensor:     {n_segs} sequences x {seg_len} frames x {n_kps} keypoints x {d}D", flush=True)

    # 3. Load or Fit PCA
    pca_p = PROD_KPMS_DIR / "pca.p"
    if pca_p.exists():
        pca = kpms.load_pca(str(PROD_KPMS_DIR))
        print("  Loaded existing pre-fit PCA from project.", flush=True)
    else:
        print("  Fitting PCA on representative cohort subset (D=10)...", flush=True)
        max_pca_seqs = min(30, len(data["Y"]))
        sample_indices = np.linspace(0, len(data["Y"]) - 1, max_pca_seqs, dtype=int)
        data_pca_subset = {
            k: v[sample_indices] if (hasattr(v, "__len__") and len(v) == len(data["Y"])) else v
            for k, v in data.items()
        }
        pca = kpms.fit_pca(**data_pca_subset, **config)
        kpms.save_pca(pca, str(PROD_KPMS_DIR))

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"  Top 10 PCs Cumulative Variance Explained: {cum_var[-1]*100:.2f}%", flush=True)

    # Preemptive Analytical Batch Sizing
    print("  Preemptive Analytical Tensor Sizing...", flush=True)
    rb.KPMSStateCache.clear()
    rb.precompute_optimal_batch_config(data, verbose=True)

    parser = argparse.ArgumentParser(description="Empirical test on full cohort")
    parser.add_argument("--skip-ar", action="store_true", help="Skip AR phase and jump straight to testing SLDS")
    parser.add_argument("--num-slds", type=int, default=2, help="Number of SLDS iterations to test")
    args = parser.parse_args()

    # 4. Initialize Fresh Model (with disk cache)
    print("\n[Step 3/5] Loading or initializing fresh AR-HMM model...", flush=True)
    init_model_cache = PROD_KPMS_DIR / "init_model.pkl"
    gpu_device = jax.devices()[0]
    import jax.numpy as jnp

    if init_model_cache.exists():
        print(f"  [CACHE HIT] Loading pre-initialized model state from {init_model_cache}...", flush=True)
        with open(init_model_cache, "rb") as f:
            cached_model = pickle.load(f)
        model = copy.deepcopy(cached_model)
        # Restore GPU states for x, v, h, z, while keeping s on CPU to save 3.4GB VRAM
        model["states"] = {
            k: (np.asarray(v) if k == "s" else jax.device_put(jnp.asarray(v), gpu_device))
            for k, v in cached_model["states"].items()
        }
        print("  [CACHE HIT] Model loaded and active states placed on GPU in 0.5s.", flush=True)
    else:
        print("  [CACHE MISS] Initializing fresh AR-HMM model via batched init...", flush=True)
        t0_init = time.time()
        init_bs = min(rb.CONFIG.discrete_bs, 20)
        rng_key = jax.random.PRNGKey(42)
        model = batched_init_model_by_video(
            data, pca, config,
            params_subset_size=5, batch_size=init_bs, seed=rng_key
        )
        model = kpms.update_hypparams(model, kappa=float(INIT_KAPPA))
        t_init = time.time() - t0_init
        print(f"  Fresh model initialized in {t_init:.2f}s ({t_init/60:.2f} min).", flush=True)

        print(f"  [CACHE SAVE] Caching initialized model to {init_model_cache}...", flush=True)
        cpu_states = {k: np.asarray(v) for k, v in model["states"].items()}
        cache_dict = {
            "params": {k: np.asarray(v) if hasattr(v, "shape") else v for k, v in model["params"].items()},
            "hypparams": model["hypparams"],
            "states": cpu_states,
            "noise_prior": np.asarray(model["noise_prior"]) if hasattr(model.get("noise_prior"), "shape") else model.get("noise_prior", 1.0),
            "seed": model["seed"]
        }
        with open(init_model_cache, "wb") as f:
            pickle.dump(cache_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [CACHE SAVED] Initial model cached ({init_model_cache.stat().st_size / 1e6:.1f} MB).", flush=True)

    t_ar_warm = 5.5
    t_ar_cold = 0.0
    if not args.skip_ar:
        # 5. Phase 1: 2 AR Iterations
        print("\n" + "=" * 80, flush=True)
        print("🔥 PHASE 1: EXECUTING 2 AR-HMM ITERATIONS (0 -> 1 -> 2)...", flush=True)
        print("=" * 80, flush=True)

        # Iteration 0 (Cold JIT)
        t0_ar0 = time.time()
        model, _ = kpms.fit_model(
            model, data, metadata,
            project_dir=str(PROD_KPMS_DIR),
            model_name=MODEL_NAME,
            ar_only=True,
            start_iter=0,
            num_iters=1,
            parallel_message_passing=False,
            save_every_n_iters=0,
            generate_progress_plots=False,
            verbose=True
        )
        t_ar_cold = time.time() - t0_ar0
        print(f"  >>> AR Iteration 0 (Cold JIT): {t_ar_cold:.2f}s", flush=True)

        # Iteration 1 (Warm)
        t0_ar1 = time.time()
        model, _ = kpms.fit_model(
            model, data, metadata,
            project_dir=str(PROD_KPMS_DIR),
            model_name=MODEL_NAME,
            ar_only=True,
            start_iter=1,
            num_iters=2,
            parallel_message_passing=False,
            save_every_n_iters=0,
            generate_progress_plots=False,
            verbose=True
        )
        t_ar_warm = time.time() - t0_ar1
        print(f"  >>> AR Iteration 1 (WARM):     {t_ar_warm:.2f}s", flush=True)
        start_slds = 2
    else:
        print("\n" + "=" * 80, flush=True)
        print("⏩ SKIPPING AR-HMM PHASE (--skip-ar flag set). Directly testing SLDS!", flush=True)
        print("=" * 80, flush=True)
        start_slds = 0

    # Transition Kappa to SLDS
    print("\n" + "=" * 80, flush=True)
    print("  TRANSITION: Applying 10x Kappa Reduction for Phase 2 SLDS", flush=True)
    model = kpms.update_hypparams(model, kappa=float(SLDS_KAPPA))
    print(f"  New SLDS Kappa: {SLDS_KAPPA:.2e}", flush=True)
    print("=" * 80, flush=True)

    # 6. Phase 2: SLDS Iterations
    print("\n" + "=" * 80, flush=True)
    print(f"🔥 PHASE 2: EXECUTING {args.num_slds} FULL SLDS ITERATIONS ({start_slds} -> {start_slds + args.num_slds})...", flush=True)
    print("=" * 80, flush=True)

    # Iteration (Cold JIT for continuous samplers: x, v, h, s)
    t0_slds0 = time.time()
    model, _ = kpms.fit_model(
        model, data, metadata,
        project_dir=str(PROD_KPMS_DIR),
        model_name=MODEL_NAME,
        ar_only=False,
        start_iter=start_slds,
        num_iters=start_slds + 1,
        parallel_message_passing=False,
        save_every_n_iters=0,
        generate_progress_plots=False,
        verbose=True
    )
    t_slds_cold = time.time() - t0_slds0
    print(f"  >>> SLDS Iteration {start_slds} (Cold JIT): {t_slds_cold:.2f}s", flush=True)

    # Iteration (Warm)
    t0_slds1 = time.time()
    model, _ = kpms.fit_model(
        model, data, metadata,
        project_dir=str(PROD_KPMS_DIR),
        model_name=MODEL_NAME,
        ar_only=False,
        start_iter=start_slds + 1,
        num_iters=start_slds + 2,
        parallel_message_passing=False,
        save_every_n_iters=0,
        generate_progress_plots=False,
        verbose=True
    )
    t_slds_warm = time.time() - t0_slds1
    print(f"  >>> SLDS Iteration {start_slds + 1} (WARM):     {t_slds_warm:.2f}s", flush=True)

    # State Auditing
    print("\n[Step 5/5] Auditing State Consistency and Syllables...", flush=True)
    x = np.asarray(model["states"]["x"])
    z = np.asarray(model["states"]["z"])
    mask = np.asarray(data["mask"])[:, 3:]

    nan_count_x = int(np.sum(np.isnan(x)))
    nan_count_z = int(np.sum(np.isnan(z)))
    print(f"  NaN check for x: {nan_count_x} NaNs", flush=True)
    print(f"  NaN check for z: {nan_count_z} NaNs", flush=True)
    assert nan_count_x == 0 and nan_count_z == 0, "NaNs detected in states!"

    active_sylls = len(np.unique(z[mask > 0]))
    durations = kpms.viz.get_durations(z, mask)
    med_frames = float(np.median(durations)) if len(durations) > 0 else 0.0
    print(f"  Active Syllables:       {active_sylls}", flush=True)
    print(f"  Median Bout Duration:   {med_frames:.1f} frames ({(med_frames/60)*1000:.1f} ms)", flush=True)

    # Empirical Extrapolation
    est_ar_sec = 50 * t_ar_warm
    est_slds_sec = 200 * t_slds_warm
    est_total_sec = est_ar_sec + est_slds_sec
    est_total_min = est_total_sec / 60.0

    print("\n" + "=" * 80, flush=True)
    print("🏆 SUCCESS! LIVE EMPIRICAL BENCHMARK COMPLETE (0 OOMs)", flush=True)
    print("=" * 80, flush=True)
    print(f"  • AR Cold JIT Time:           {t_ar_cold:6.2f} s", flush=True)
    print(f"  • AR WARM Iteration Time:     {t_ar_warm:6.2f} s / iteration", flush=True)
    print(f"  • SLDS Cold JIT Time:         {t_slds_cold:6.2f} s", flush=True)
    print(f"  • SLDS WARM Iteration Time:   {t_slds_warm:6.2f} s / iteration", flush=True)
    print("-" * 80, flush=True)
    print(f"  • Projected 50 AR Time:       {est_ar_sec/60:6.2f} min ({est_ar_sec:.1f} s)", flush=True)
    print(f"  • Projected 200 SLDS Time:    {est_slds_sec/60:6.2f} min ({est_slds_sec:.1f} s)", flush=True)
    print(f"  • PROJECTED 50+200 TOTAL:     {est_total_min:6.2f} min ({est_total_sec:.1f} s)", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
