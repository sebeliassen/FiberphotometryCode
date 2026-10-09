#!/usr/bin/env python3
"""
run_tandem_cohort_production_50ar_200slds.py
=============================================================================
Full Cohort Production Tandem (Multi-View) Keypoint-MoSeq Training:
50 AR Iterations + 200 SLDS Iterations (250 Total Gibbs Iterations)
across ALL 118 Available Tandem Sessions (24,111,516 frames @ 60 FPS).

Features:
1. Resilient & Detached: Checkpoints every 25 iterations with automatic resume.
2. Zero-OOM & Peak Speed: Preemptive analytical tensor sizing and Amdahl 27x
   Banded Cholesky JAX kernels on NVIDIA RTX 4090 (24.5 GB VRAM).
3. Real-Time Telemetry: Live unbuffered stdout + ntfy alerts at start, every 25
   iterations (with duration distributions & ETA), phase boundary, and completion.
4. Kinematics & Photometry Weaving:
   - Full 23-keypoint latent pose trajectories (x, z).
   - TopCam native and BotCam native (972x972) kinematics & velocities.
   - Photometry streams (dFF, z-score, Wallace, RANSAC).
   - Serialized cleanly to sessions_karen_project_tandem_cohort_production.pkl.
=============================================================================
"""

import os
import sys

# Live unbuffered stdout/stderr
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Hardware environment flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["USE_MEMORY_PATCH"] = "1"

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import copy
import pickle
import shutil
import signal
import time
import traceback
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import jax
import keypoint_moseq as kpms
import pandas as pd
from jax_moseq.utils.debugging import convert_data_precision
from scipy.ndimage import median_filter

from fpcode.common.paths import REPO_ROOT
KPMS_UTILS_DIR = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll"
if str(KPMS_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(KPMS_UTILS_DIR))

import utils.resample_batching as rb
from utils.batch_init import batched_init_model_by_video
from utils.trimming import filter_with_masks, presence_mask_from_conf

from syllable_generation.analysis.replications.datta_utils import combined_photometry_and_syllables
from syllable_generation.analysis.utils.sampling import flatten_results_to_timeline
from syllable_generation.multiview_top_bot_experiments.audit_cohort_correlations import find_matched_pairs
from syllable_generation.multiview_top_bot_experiments.core.ttl_sync import get_session_ttl_offset

# Configuration constants
EXPERIMENT_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
FUSED_DIR = EXPERIMENT_DIR / "fused_h5_full_cohort"
DIRTY_PKL_IN = REPO_ROOT / "syllable_generation/data/sessions_karen_project_dirty.pkl"
K240K_PKL_REF = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_k240k.pkl"
OUTPUT_PROD_PKL = EXPERIMENT_DIR / "sessions_karen_project_tandem_cohort_production.pkl"

PROD_KPMS_DIR = EXPERIMENT_DIR / "kpms_tandem_cohort_production"
PROD_DATA_DIR = PROD_KPMS_DIR / "data"
MODEL_NAME = "tandem_cohort_118sess_50ar_200slds"

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

NUM_AR_ITERS = 50
NUM_SLDS_ITERS = 200
TOTAL_ITERS = NUM_AR_ITERS + NUM_SLDS_ITERS
SAVE_EVERY_N_ITERS = 25

INIT_KAPPA = 7.5e13
DECREASE_KAPPA_FACTOR = 10.0
SLDS_KAPPA = INIT_KAPPA / DECREASE_KAPPA_FACTOR
TARGET_FPS = 60

from fpcode.common.notify import send_ntfy

def handle_shutdown_signal(signum, frame):
    sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"\n[shutdown] Caught signal {sig_name} (PID {os.getpid()}). Flushing and alerting...", flush=True)
    send_ntfy(
        f"KPMS Tandem Run Interrupted ({sig_name})",
        f"Process caught {sig_name}. Resuming this script will continue from the latest checkpoint.",
        priority="high",
        tags="warning,pause"
    )
    sys.exit(128 + signum)

signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full Cohort Tandem KPMS Production Training")
    parser.add_argument("--init-kappa", type=float, default=INIT_KAPPA, help="Initial Kappa for AR phase")
    parser.add_argument("--decrease-kappa-factor", type=float, default=DECREASE_KAPPA_FACTOR, help="Reduction factor for SLDS phase")
    parser.add_argument("--num-ar-iters", type=int, default=NUM_AR_ITERS, help="Number of AR iters")
    parser.add_argument("--num-slds-iters", type=int, default=NUM_SLDS_ITERS, help="Number of SLDS iters")
    parser.add_argument("--save-every-n-iters", type=int, default=SAVE_EVERY_N_ITERS, help="Checkpoint save interval")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model initialization")
    parser.add_argument("--output-pkl", type=str, default=str(OUTPUT_PROD_PKL), help="Target output pickle path")
    args = parser.parse_args()

    init_kappa = args.init_kappa
    decrease_kappa_factor = args.decrease_kappa_factor
    slds_kappa = init_kappa / decrease_kappa_factor
    num_ar_iters = args.num_ar_iters
    num_slds_iters = args.num_slds_iters
    total_iters = num_ar_iters + num_slds_iters
    save_every_n_iters = args.save_every_n_iters
    model_name = args.model_name
    seed = args.seed
    output_prod_pkl = Path(args.output_pkl)

    t_program_start = time.time()
    pid = os.getpid()

    # Discover all matched pairs
    pairs = find_matched_pairs()
    pair_map = {p["sid"]: p for p in pairs}
    print("=" * 80, flush=True)
    print(f"🚀 TANDEM KPMS FULL COHORT PRODUCTION RUN: {len(pairs)} SESSIONS", flush=True)
    print(f"  Plan:                {num_ar_iters} AR + {num_slds_iters} SLDS = {total_iters} TOTAL ITERS", flush=True)
    print(f"  Date & Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  Process PID:         {pid}", flush=True)
    print(f"  Model Name:          {model_name}", flush=True)
    print(f"  Random Seed:         {seed}", flush=True)
    print("  Hardware:            RTX 4090 (24.5 GB VRAM) with Amdahl 27x Optimizations", flush=True)
    print(f"  Init Kappa:          {init_kappa:.2e} -> SLDS Kappa: {slds_kappa:.2e}", flush=True)
    print(f"  Save Checkpoints:    Every {save_every_n_iters} iterations", flush=True)
    print(f"  Project Directory:   {PROD_KPMS_DIR}", flush=True)
    print(f"  Target Pickle:       {output_prod_pkl}", flush=True)
    print("=" * 80, flush=True)

    # 1. Apply high-performance kernels
    print("\n[Step 1/6] Applying Amdahl 27x Optimizations (Banded Cholesky + GPU Kernels)...", flush=True)
    rb.apply_optimizations()

    # Send launch notification
    send_ntfy(
        f"KPMS Tandem Cohort Run Started ({model_name}) 🚀",
        f"PID: {pid}\n"
        f"Plan: {num_ar_iters} AR + {num_slds_iters} SLDS = {total_iters} iters\n"
        f"Sessions: {len(pairs)} tandem sessions (~24.1M frames @ 60 FPS)\n"
        f"Kappa: {init_kappa:.2e} -> {slds_kappa:.2e}\n"
        f"Checkpoints every {save_every_n_iters} iters",
        priority="default",
        tags="rocket,brain"
    )

    # 2. Ingest Base Sessions & Enrich
    print("\n[Step 2/6] Loading base SessionRecord objects from dirty pickle...", flush=True)
    with open(DIRTY_PKL_IN, "rb") as f:
        dirty_sessions = pickle.load(f)

    selected_sessions = [copy.deepcopy(s) for s in dirty_sessions if s.session_id in pair_map]
    print(f"  Loaded {len(selected_sessions)} matching SessionRecord objects.")

    for s in selected_sessions:
        sid = s.session_id
        cfg = pair_map[sid]
        fused_path = FUSED_DIR / f"{sid}_tandem_23kp.h5"

        # Tracking refs enrichment
        t_df = s.tracking_refs_df.copy()
        if not (t_df["role"] == "ventral").any():
            ventral_track_row = {
                "path": str(cfg["bot_h5"]),
                "path_type": "tracking",
                "kind": "h5",
                "role": "ventral",
                "camset": "BotCamVid",
                "cam_slot": "bottom_cam",
                "arena": cfg["arena"],
                "chamber": s.metadata.get("chamber"),
                "ts": s.metadata.get("timestamp"),
                "camera_key": f"bottom_cam:{s.metadata.get('chamber')}",
                "chamber_index": s.metadata.get("chamber_index", 0)
            }
            t_df = pd.concat([t_df, pd.DataFrame([ventral_track_row])], ignore_index=True)

        if not (t_df["role"] == "fused_23kp").any():
            fused_track_row = {
                "path": str(fused_path),
                "path_type": "tracking",
                "kind": "h5",
                "role": "fused_23kp",
                "camset": "TandemMultiview",
                "cam_slot": "multiview_fused",
                "arena": cfg["arena"],
                "chamber": s.metadata.get("chamber"),
                "ts": s.metadata.get("timestamp"),
                "camera_key": f"multiview_fused:{s.metadata.get('chamber')}",
                "chamber_index": s.metadata.get("chamber_index", 0)
            }
            t_df = pd.concat([t_df, pd.DataFrame([fused_track_row])], ignore_index=True)
        s.tracking_refs_df = t_df

    # 3. Setup KPMS Project & Data Directory
    print("\n[Step 3/6] Setting up KPMS production project environment...", flush=True)
    PROD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    fused_files = sorted(FUSED_DIR.glob("*.h5"))
    print(f"  Symlinking {len(fused_files)} fused H5 files into {PROD_DATA_DIR}...")
    for f in fused_files:
        dst = PROD_DATA_DIR / f.name
        if not dst.exists():
            dst.symlink_to(f)

    config_source = EXPERIMENT_DIR / "multiview_kappa_search/project/config.yml"
    config_dest = PROD_KPMS_DIR / "config.yml"
    if not config_dest.exists():
        shutil.copy(config_source, config_dest)

    config = kpms.load_config(str(PROD_KPMS_DIR))
    config["fps"] = TARGET_FPS
    config["trans_hypparams"]["kappa"] = init_kappa
    kpms.update_config(str(PROD_KPMS_DIR), **config)

    # Multi-tier disk caching: Load or compute preprocessed tracking data
    preprocessed_cache = PROD_KPMS_DIR / "preprocessed_data.pkl"
    if preprocessed_cache.exists():
        print(f"  [CACHE HIT] Loading preprocessed cohort tracking data from {preprocessed_cache}...", flush=True)
        with open(preprocessed_cache, "rb") as f:
            data, metadata = pickle.load(f)
    else:
        print("  [CACHE MISS] Loading 23-keypoint tracking data into KPMS format...", flush=True)
        coordinates, confidences, bodyparts = kpms.load_keypoints(str(PROD_DATA_DIR), format="deeplabcut")

        # 1. Automated Machine Exclusion via Presence Masks (source of auto_mask.npz)
        print("  Computing presence masks (conf >= 0.9, min fraction parts >= 0.4)...", flush=True)
        all_masks = {name: presence_mask_from_conf(conf) for name, conf in confidences.items()}

        # 2. Filter with masks & outlier removal (generates video_frame_indexes)
        print("  Filtering coordinates with presence masks and removing outliers...", flush=True)
        filtered_coords, filtered_confs, video_frame_indexes = filter_with_masks(
            coordinates, confidences, all_masks, str(PROD_KPMS_DIR),
            return_video_frame_indexes=True, **config
        )

        # Automatically remove any sequence with < 5 frames (e.g. empty chambers)
        keys_to_remove = [k for k, v in filtered_coords.items() if len(v) < 5]
        for k in keys_to_remove:
            print(f"  [Auto-Exclusion] Dropping '{k}' ({len(filtered_coords[k])} frames present, requires >= 5).", flush=True)
            filtered_coords.pop(k)
            filtered_confs.pop(k)
            if k in video_frame_indexes:
                video_frame_indexes.pop(k)

        # Save video_frame_indexes.npz and create auto_mask.npz symlink
        npz_out = PROD_KPMS_DIR / "video_frame_indexes.npz"
        np.savez_compressed(
            npz_out,
            **{k: np.asarray(v, dtype=np.int64) for k, v in video_frame_indexes.items()}
        )
        auto_mask_sym = PROD_KPMS_DIR / "auto_mask.npz"
        if auto_mask_sym.exists() or auto_mask_sym.is_symlink():
            auto_mask_sym.unlink()
        auto_mask_sym.symlink_to(npz_out)
        print(f"  Saved video_frame_indexes.npz and auto_mask.npz ({len(video_frame_indexes)} valid sessions).", flush=True)

        data, metadata = kpms.format_data(filtered_coords, filtered_confs, **config)
        data = convert_data_precision(data, np.float64)
        data["Y"] = np.asarray(data["Y"])

        print(f"  [CACHE SAVE] Caching preprocessed cohort data to {preprocessed_cache}...", flush=True)
        with open(preprocessed_cache, "wb") as f:
            pickle.dump((data, metadata), f, protocol=pickle.HIGHEST_PROTOCOL)

    total_frames = int(data["mask"].sum())
    n_segs, seg_len, n_kps, d = data["Y"].shape
    print(f"  Active animal frames: {total_frames:,}")
    print(f"  Batched Y tensor:     {n_segs} sequences x {seg_len} frames x {n_kps} keypoints x {d}D")

    # Load or fit PCA
    pca_p = PROD_KPMS_DIR / "pca.p"
    pca_asdf = PROD_KPMS_DIR / "pca.asdf"
    if pca_p.exists() or pca_asdf.exists():
        pca = kpms.load_pca(str(PROD_KPMS_DIR))
        print("  Loaded existing PCA from project.", flush=True)
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
    print(f"  Top 10 PCs Cumulative Variance Explained: {cum_var[-1]*100:.2f}%")

    # Preemptive Analytical Batch Sizing
    print("  Executing Preemptive Analytical Tensor Sizing...", flush=True)
    rb.KPMSStateCache.clear()
    rb.precompute_optimal_batch_config(data, verbose=True)

    # 4. Checkpoint Resumption or Clean Initialization
    checkpoint_file = PROD_KPMS_DIR / model_name / "checkpoint.h5"
    current_iter = 0
    model = None

    if checkpoint_file.exists():
        try:
            print(f"\n  [RESUME] Found existing checkpoint at {checkpoint_file}. Resuming...", flush=True)
            model, data, metadata, current_iter = kpms.load_checkpoint(str(PROD_KPMS_DIR), model_name)
            data = convert_data_precision(data, np.float64)
            data["Y"] = np.asarray(data["Y"])
            rb.KPMSStateCache.clear()
            rb.precompute_optimal_batch_config(data, model=model, verbose=True)
            print(f"  [RESUME] Resumed successfully from iteration {current_iter}.", flush=True)
            send_ntfy(
                "KPMS Resumed from Checkpoint",
                f"Resumed training '{model_name}' at iteration {current_iter}/{total_iters}.",
                priority="default",
                tags="play,brain"
            )
        except Exception as e:
            print(f"  [warning] Failed to load checkpoint ({e}). Initializing fresh model...", flush=True)
            model = None
            current_iter = 0

    if model is None:
        init_model_cache = PROD_KPMS_DIR / f"init_model_seed_{seed}.pkl" if seed != 42 else PROD_KPMS_DIR / "init_model.pkl"
        gpu_device = jax.devices()[0]
        import jax.numpy as jnp

        if init_model_cache.exists():
            print(f"\n  [CACHE HIT] Loading pre-initialized model state from {init_model_cache} (Seed {seed})...", flush=True)
            t0_cache = time.time()
            with open(init_model_cache, "rb") as f:
                cached_model = pickle.load(f)
            model = copy.deepcopy(cached_model)
            model["states"] = {
                k: (np.asarray(v) if k == "s" else jax.device_put(jnp.asarray(v), gpu_device))
                for k, v in cached_model["states"].items()
            }
            model = kpms.update_hypparams(model, kappa=float(init_kappa))
            t_cache = time.time() - t0_cache
            print(f"  [CACHE HIT] Model loaded and states placed on GPU in {t_cache:.2f}s.", flush=True)
        else:
            print(f"\n  [INIT] Initializing fresh AR-HMM model with batched init (Seed {seed})...", flush=True)
            t0_init = time.time()
            init_bs = min(rb.CONFIG.discrete_bs, 20)
            rng_key = jax.random.PRNGKey(seed)
            model = batched_init_model_by_video(
                data, pca, config,
                params_subset_size=5, batch_size=init_bs, seed=rng_key
            )
            model = kpms.update_hypparams(model, kappa=float(init_kappa))
            t_init = time.time() - t0_init
            print(f"  [INIT] Initialization complete in {t_init:.2f}s ({t_init/60:.2f} min).", flush=True)

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

        current_iter = 0

    # 5. Model Fitting Loop
    print(f"\n[Step 4/6] Executing KPMS Training Loop ({num_ar_iters} AR + {num_slds_iters} SLDS)...", flush=True)

    # PHASE 1: AR-ONLY FITTING (0 -> 50)
    if current_iter < num_ar_iters:
        print("\n" + "=" * 70, flush=True)
        print(f"  PHASE 1: AR-Only Fitting (iter {current_iter} -> {num_ar_iters})", flush=True)
        print(f"  Kappa: {init_kappa:.2e}", flush=True)
        print("=" * 70, flush=True)

        ar_start_wall = time.time()
        chunk_size = save_every_n_iters

        while current_iter < num_ar_iters:
            next_target = min(current_iter + chunk_size, num_ar_iters)
            chunk_t0 = time.time()
            print(f"\n>>> Running AR Chunk: iter {current_iter} -> {next_target}...", flush=True)

            model, _ = kpms.fit_model(
                model, data, metadata,
                project_dir=str(PROD_KPMS_DIR),
                model_name=model_name,
                ar_only=True,
                start_iter=current_iter,
                num_iters=next_target,
                parallel_message_passing=False,
                save_every_n_iters=save_every_n_iters,
                generate_progress_plots=True,
                verbose=True
            )
            current_iter = next_target

            # Syllable duration diagnostics
            z_curr = np.array(model["states"]["z"])
            mask_curr = np.array(data["mask"])[:, 3:]
            dur_list = kpms.viz.get_durations(z_curr, mask_curr)
            med_frames = float(np.median(dur_list)) if len(dur_list) > 0 else 0.0
            med_ms = (med_frames / TARGET_FPS) * 1000.0
            mean_frames = float(np.mean(dur_list)) if len(dur_list) > 0 else 0.0
            n_active_sylls = len(np.unique(z_curr[mask_curr > 0]))

            sec_per_it = (time.time() - ar_start_wall) / current_iter
            ar_rem_sec = (num_ar_iters - current_iter) * sec_per_it
            eta_str = str(timedelta(seconds=int(ar_rem_sec)))

            print(f"--- Chunk Complete: AR {current_iter}/{num_ar_iters} ({sec_per_it:.2f}s/it, AR ETA: {eta_str}) | Median: {med_frames:.1f}f ({med_ms:.1f}ms) | Mean: {mean_frames:.1f}f | Sylls: {n_active_sylls} ---", flush=True)

            send_ntfy(
                f"KPMS AR Progress: {current_iter}/{num_ar_iters}",
                f"AR {current_iter}/{num_ar_iters} complete ({(current_iter/num_ar_iters)*100:.0f}%)\n"
                f"Median Syllable: {med_frames:.1f} frames ({med_ms:.1f} ms) | Active: {n_active_sylls}\n"
                f"Speed: {sec_per_it:.2f}s/it | AR ETA: {eta_str}",
                priority="default",
                tags="chart_with_upwards_trend"
            )

        ar_total_min = (time.time() - ar_start_wall) / 60.0
        print(f"\n[PHASE 1 COMPLETE] {num_ar_iters} AR iterations finished in {ar_total_min:.1f} min.", flush=True)
        send_ntfy(
            f"KPMS Phase 1 ({num_ar_iters} AR) Complete 🎉",
            f"{num_ar_iters} AR iterations finished in {ar_total_min:.1f} min.\n"
            f"Transitioning to Phase 2 SLDS (10x kappa reduction: {slds_kappa:.1e}).",
            priority="default",
            tags="tada,repeat"
        )
    else:
        print(f"  [Phase 1] Skipping AR (Already reached iter {current_iter} >= {num_ar_iters}).", flush=True)

    # TRANSITION: 10x Kappa reduction for SLDS
    print("\n" + "=" * 70, flush=True)
    print("  TRANSITION: Applying 10x Kappa Reduction for Phase 2 SLDS", flush=True)
    model = kpms.update_hypparams(model, kappa=float(slds_kappa))
    print(f"  New Kappa: {slds_kappa:.2e}", flush=True)
    print("=" * 70, flush=True)

    # PHASE 2: SLDS FITTING
    if current_iter < total_iters:
        print("\n" + "=" * 70, flush=True)
        print(f"  PHASE 2: Full SLDS Fitting (iter {current_iter} -> {total_iters})", flush=True)
        print(f"  SLDS Kappa: {slds_kappa:.2e}", flush=True)
        print("=" * 70, flush=True)

        slds_start_wall = time.time()
        slds_start_iter = current_iter
        chunk_size = save_every_n_iters

        while current_iter < total_iters:
            next_target = min(current_iter + chunk_size, total_iters)
            chunk_t0 = time.time()
            print(f"\n>>> Running SLDS Chunk: iter {current_iter} -> {next_target}...", flush=True)

            model, _ = kpms.fit_model(
                model, data, metadata,
                project_dir=str(PROD_KPMS_DIR),
                model_name=model_name,
                ar_only=False,
                start_iter=current_iter,
                num_iters=next_target,
                parallel_message_passing=False,
                save_every_n_iters=save_every_n_iters,
                generate_progress_plots=True,
                verbose=True
            )
            current_iter = next_target

            z_curr = np.array(model["states"]["z"])
            mask_curr = np.array(data["mask"])[:, 3:]
            dur_list = kpms.viz.get_durations(z_curr, mask_curr)
            med_frames = float(np.median(dur_list)) if len(dur_list) > 0 else 0.0
            med_ms = (med_frames / TARGET_FPS) * 1000.0
            mean_frames = float(np.mean(dur_list)) if len(dur_list) > 0 else 0.0
            n_active_sylls = len(np.unique(z_curr[mask_curr > 0]))

            slds_done = current_iter - num_ar_iters
            sec_per_it = (time.time() - slds_start_wall) / (current_iter - slds_start_iter)
            slds_rem_sec = (total_iters - current_iter) * sec_per_it
            eta_str = str(timedelta(seconds=int(slds_rem_sec)))
            est_time = (datetime.now() + timedelta(seconds=slds_rem_sec)).strftime('%H:%M:%S')

            print(f"--- Chunk Complete: SLDS {slds_done}/{num_slds_iters} (Total {current_iter}/{total_iters}) | {sec_per_it:.2f}s/it | Median: {med_frames:.1f}f ({med_ms:.1f}ms) | Mean: {mean_frames:.1f}f | Sylls: {n_active_sylls} | ETA: {eta_str} (Est: {est_time}) ---", flush=True)

            send_ntfy(
                f"KPMS SLDS Progress: {slds_done}/{num_slds_iters}",
                f"SLDS {slds_done}/{num_slds_iters} complete (Total {current_iter}/{total_iters}, {(current_iter/total_iters)*100:.0f}%)\n"
                f"Median Syllable: {med_frames:.1f} frames ({med_ms:.1f} ms) | Active: {n_active_sylls}\n"
                f"Speed: {sec_per_it:.2f}s/it | ETA: {eta_str} (Est: {est_time})",
                priority="default",
                tags="hourglass_flowing_sand,brain"
            )

        slds_total_min = (time.time() - slds_start_wall) / 60.0
        print(f"\n[PHASE 2 COMPLETE] {num_slds_iters} SLDS iterations finished in {slds_total_min:.1f} min.", flush=True)
    else:
        print(f"  [Phase 2] Training already complete ({current_iter} >= {total_iters}).", flush=True)

    # 6. Extract Results and Weave Kinematics & Photometry
    print("\n[Step 5/6] Extracting final model results and weaving tandem kinematics...", flush=True)
    model, data, metadata, current_iter = kpms.load_checkpoint(str(PROD_KPMS_DIR), model_name)
    results = kpms.extract_results(model, metadata, str(PROD_KPMS_DIR), model_name, save_results=True)

    # Load video_frame_indexes for exact timeline flattening
    vfi_path = PROD_KPMS_DIR / "video_frame_indexes.npz"
    vfi_dict = {}
    if vfi_path.exists():
        try:
            with np.load(vfi_path) as z:
                vfi_dict = {k: z[k].astype(int) for k in z.files}
            print(f"  Loaded VFI mappings for {len(vfi_dict)} sessions from {vfi_path.name}.")
        except Exception as e:
            print(f"  [warning] Could not load VFI from npz: {e}", flush=True)

    k240k_map = {}
    if K240K_PKL_REF.exists():
        try:
            with open(K240K_PKL_REF, "rb") as f:
                k240k_all = pickle.load(f)
            for s_ref in k240k_all:
                if s_ref.session_id in pair_map:
                    k240k_map[s_ref.session_id] = s_ref
        except Exception as e:
            print(f"  [warning] Could not load K240K reference: {e}", flush=True)

    modeled_sessions = []
    for s in selected_sessions:
        sid = s.session_id
        cfg = pair_map[sid]
        rec_key = f"{sid}_tandem_23kp"
        s.results_dict_key = rec_key
        if rec_key not in results:
            print(f"  [skip] {rec_key} not in results dictionary.", flush=True)
            continue
        res = results[rec_key]

        df_top = pd.read_hdf(cfg["top_h5"])
        df_bot = pd.read_hdf(cfg["bot_h5"])
        top_scorer = df_top.columns.levels[0][0]
        bot_scorer = df_bot.columns.levels[0][0]

        N_top = len(df_top)
        N_bot = len(df_bot)
        timeline_index = pd.RangeIndex(0, N_top)

        # 1. Populate s.syllables via VFI timeline flattening (Golden Rule: TopCam Master Timeline)
        if rec_key in vfi_dict:
            flat_res = flatten_results_to_timeline(
                {rec_key: res},
                {rec_key: vfi_dict[rec_key]},
                actual_lengths={rec_key: N_top}
            )[rec_key]
            s.syllables = flat_res
        else:
            tandem_sylls = res["syllable"]
            syll_series = pd.Series(pd.NA, index=timeline_index, dtype="Int64")
            syll_series.iloc[:min(N_top, len(tandem_sylls))] = pd.Series(tandem_sylls[:min(N_top, len(tandem_sylls))], dtype="Int64")
            s.syllables = pd.DataFrame({"syllable": syll_series}, index=timeline_index)

        if "syllable" in s.syllables.columns:
            s.syllables["syllable"] = s.syllables["syllable"].astype("Int64")

        # 2. Extract TopCam Raw Kinematics (native timeline 0..N_top-1)
        top_nx = df_top[top_scorer, "nose", "x"].to_numpy(dtype=float)
        top_ny = df_top[top_scorer, "nose", "y"].to_numpy(dtype=float)
        top_tx = df_top[top_scorer, "tail_base", "x"].to_numpy(dtype=float)
        top_ty = df_top[top_scorer, "tail_base", "y"].to_numpy(dtype=float)

        cx_top = 0.5 * (top_nx + top_tx)
        cy_top = 0.5 * (top_ny + top_ty)
        heading_top = np.arctan2(top_ny - top_ty, top_nx - top_tx)

        dx_top = np.diff(cx_top)
        dy_top = np.diff(cy_top)
        vel_top_raw = np.pad(np.sqrt(dx_top**2 + dy_top**2) * 60.0, (0, 1), mode="edge")
        ang_vel_top = np.pad(np.diff(np.unwrap(heading_top)) * 60.0, (0, 1), mode="edge")
        vel_top_smooth = median_filter(np.clip(vel_top_raw, 0, 300.0), size=7, mode="nearest")

        # 3. Extract BotCam Raw Kinematics (TTL hardware-aligned to TopCam timeline)
        offset_info = get_session_ttl_offset(sid)
        offset = int(offset_info.get("bot0_in_top", 0) - offset_info.get("top0_in_bot", 0))

        t_start = max(0, offset)
        t_end = min(N_top, N_bot + offset)
        top_overlap = np.arange(t_start, t_end)
        bot_overlap = top_overlap - offset

        bot_nx_raw = df_bot[bot_scorer, "nose_tip", "x"].to_numpy(dtype=float)
        bot_ny_raw = df_bot[bot_scorer, "nose_tip", "y"].to_numpy(dtype=float)
        bot_tx_raw = df_bot[bot_scorer, "tail_base", "x"].to_numpy(dtype=float)
        bot_ty_raw = df_bot[bot_scorer, "tail_base", "y"].to_numpy(dtype=float)

        cx_bot_native = 0.5 * (bot_nx_raw + bot_tx_raw)
        cy_bot_native = 0.5 * (bot_ny_raw + bot_ty_raw)
        heading_bot_native = np.arctan2(bot_ny_raw - bot_ty_raw, bot_nx_raw - bot_tx_raw)

        dx_bot = np.diff(cx_bot_native)
        dy_bot = np.diff(cy_bot_native)
        vel_bot_raw_native = np.pad(np.sqrt(dx_bot**2 + dy_bot**2) * 60.0, (0, 1), mode="edge")
        ang_vel_bot_native = np.pad(np.diff(np.unwrap(heading_bot_native)) * 60.0, (0, 1), mode="edge")
        vel_bot_smooth_native = median_filter(np.clip(vel_bot_raw_native, 0, 300.0), size=7, mode="nearest")

        # Project native BotCam kinematics onto TopCam timeline
        cx_bot = np.full(N_top, np.nan, dtype=float)
        cy_bot = np.full(N_top, np.nan, dtype=float)
        heading_bot = np.full(N_top, np.nan, dtype=float)
        vel_bot_smooth = np.full(N_top, np.nan, dtype=float)
        vel_bot_raw = np.full(N_top, np.nan, dtype=float)
        ang_vel_bot = np.full(N_top, np.nan, dtype=float)

        if len(top_overlap) > 0:
            cx_bot[top_overlap] = cx_bot_native[bot_overlap]
            cy_bot[top_overlap] = cy_bot_native[bot_overlap]
            heading_bot[top_overlap] = heading_bot_native[bot_overlap]
            vel_bot_smooth[top_overlap] = vel_bot_smooth_native[bot_overlap]
            vel_bot_raw[top_overlap] = vel_bot_raw_native[bot_overlap]
            ang_vel_bot[top_overlap] = ang_vel_bot_native[bot_overlap]

        # Onset calculation: True where valid syllable transitions occur
        syll_series = s.syllables["syllable"]
        syll_vals = syll_series.to_numpy()
        is_valid = ~pd.isna(syll_vals)
        diff_mask = np.zeros(N_top, dtype=bool)
        if np.any(is_valid):
            first_idx = np.where(is_valid)[0][0]
            diff_mask[first_idx] = True
            for t in range(first_idx + 1, N_top):
                if is_valid[t] and (not is_valid[t-1] or syll_vals[t] != syll_vals[t-1]):
                    diff_mask[t] = True

        moseq_heading = s.syllables["heading"].to_numpy() if "heading" in s.syllables.columns else heading_top

        kinematics_dict = {
            "frame_index": timeline_index.to_numpy(),
            "name": [rec_key] * N_top,
            "centroid_x": cx_top,
            "centroid_y": cy_top,
            "heading": moseq_heading,
            "heading_top": heading_top,
            "heading_moseq": moseq_heading,
            "angular_velocity": ang_vel_top,
            "velocity_px_s": vel_top_smooth,
            "velocity_px_s_raw": vel_top_raw,
            "syllable": syll_series,
            "tandem_syllable": syll_series,
            "group": ["default"] * N_top,
            "onset": diff_mask,
            "centroid_x_top": cx_top,
            "centroid_y_top": cy_top,
            "angular_velocity_top": ang_vel_top,
            "velocity_px_s_top": vel_top_smooth,
            "velocity_px_s_top_raw": vel_top_raw,
            "centroid_x_bot": cx_bot,
            "centroid_y_bot": cy_bot,
            "heading_bot": heading_bot,
            "angular_velocity_bot": ang_vel_bot,
            "velocity_px_s_bot": vel_bot_smooth,
            "velocity_px_s_bot_raw": vel_bot_raw,
        }

        if sid in k240k_map and "syllable" in k240k_map[sid].all_data_df.columns:
            k_syll = k240k_map[sid].all_data_df["syllable"]
            k_syll_arr = np.full(N_top, np.nan)
            k_len = min(N_top, len(k_syll))
            k_syll_arr[:k_len] = k_syll.iloc[:k_len].to_numpy()
            kinematics_dict["top_syllable"] = pd.Series(k_syll_arr, dtype="Int64")

        df_behavior = pd.DataFrame(kinematics_dict, index=timeline_index)

        # Weave Photometry (with graceful fallback for sessions lacking photometry)
        TEMPLATE_PHOT_COLS = [
            "SystemTimestamp",
            "signal_0", "signal_1", "signal_2", "signal_3",
            "signal_0_zscore_ransac", "signal_0_zscore_wallace", "signal_0_dff", "signal_0_zscore", "signal_0_prominence",
            "signal_1_zscore_ransac", "signal_1_zscore_wallace", "signal_1_dff", "signal_1_zscore", "signal_1_prominence",
            "signal_2_zscore_ransac", "signal_2_zscore_wallace", "signal_2_dff", "signal_2_zscore", "signal_2_prominence",
            "signal_3_zscore_ransac", "signal_3_zscore_wallace", "signal_3_dff", "signal_3_zscore", "signal_3_prominence"
        ]
        has_fp = hasattr(s, "photometry") and isinstance(s.photometry, dict) and "470" in s.photometry
        if has_fp:
            try:
                phot_df = combined_photometry_and_syllables(s)
                phot_aligned = phot_df.reindex(timeline_index).reset_index(drop=True)
                for col in TEMPLATE_PHOT_COLS:
                    if col not in phot_aligned.columns:
                        phot_aligned[col] = np.nan
                phot_aligned = phot_aligned[TEMPLATE_PHOT_COLS]
            except Exception as e:
                print(f"  [warning] Photometry alignment failed for {sid}: {e}. Filling with NaNs.", flush=True)
                phot_aligned = pd.DataFrame(np.nan, index=timeline_index, columns=TEMPLATE_PHOT_COLS)
        else:
            phot_aligned = pd.DataFrame(np.nan, index=timeline_index, columns=TEMPLATE_PHOT_COLS)

        s.all_data_df = pd.concat([df_behavior.reset_index(drop=True), phot_aligned], axis=1)
        modeled_sessions.append(s)

    # 7. Serialize to Output Pickle
    print(f"\n[Step 6/6] Serializing production pickle to {output_prod_pkl}...", flush=True)
    output_prod_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_prod_pkl, "wb") as f:
        pickle.dump(modeled_sessions, f)

    file_size_mb = output_prod_pkl.stat().st_size / (1024 * 1024)
    total_elapsed_min = (time.time() - t_program_start) / 60.0
    total_frames = sum(len(s.all_data_df) for s in modeled_sessions)

    print("=" * 80, flush=True)
    print(f"🎉 PRODUCTION RUN COMPLETED SUCCESSFULLY IN {total_elapsed_min:.1f} MIN!", flush=True)
    print(f"  Saved File:   {output_prod_pkl.name} ({file_size_mb:.2f} MB)", flush=True)
    print(f"  Total Frames: {total_frames:,}", flush=True)
    print("=" * 80, flush=True)

    send_ntfy(
        f"KPMS Cohort Run Succeeded! ({model_name})",
        f"Completed {num_ar_iters} AR + {num_slds_iters} SLDS in {total_elapsed_min:.1f} min.\n"
        f"Output: {output_prod_pkl.name} ({file_size_mb:.1f} MB)\n"
        f"Total frames: {total_frames:,}",
        priority="high",
        tags="white_check_mark,party_popper"
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        err_tb = traceback.format_exc()
        print("\n" + "!" * 80, flush=True)
        print(f"FATAL ERROR IN TANDEM PRODUCTION RUN:\n{err_tb}", flush=True)
        print("!" * 80, flush=True)
        send_ntfy(
            "KPMS Production Run FAILED ❌",
            f"Fatal error encountered:\n{str(exc)[:200]}\nCheck logs for traceback.",
            priority="urgent",
            tags="x,skull"
        )
        sys.exit(1)
