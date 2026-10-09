#!/usr/bin/env python3
"""
run_amph_8kp_production.py

Production runner for Keypoint-MoSeq models on the Jonathan AMPH dataset using the
8-keypoint skeleton:
  - 8 informative keypoints: nose, ears, neck, mid_back, tail_base, midsides
  - Latent Dim = 6 (95.52% EVR)
  - 50 AR iterations + 150 SLDS iterations (200 total)
  - Init Kappa = 150,000 (SLDS Kappa = 15,000 via factor of 10)
  - Checkpointing every 50 iterations (conserves disk space)
  - Memory & speed optimizations active (Banded Cholesky + CPU HMM + GPU emissions)
  - Downsampled 2x -> 40 FPS

Usage:
  python run_amph_8kp_production.py [MODEL_NAME] [SEED] [SAVE_EVERY_N_ITERS]
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import sys
import multiprocessing
if multiprocessing.current_process().name != 'MainProcess':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import logging
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    logging.getLogger("jax").setLevel(logging.CRITICAL)

sys.path.insert(0, "/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
np.bool8 = np.bool_

import json
import time
from pathlib import Path
from kpms_pipeline import run_pipeline
import pandas as pd

# ========= 8-KEYPOINT AMPH CONFIG =========
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "amph_project_8kp")
DETECTOR = "deeplabcut"
CAM = "center_cam"
SUBROUTINE = "jonathan"
LATENT_DIM = 6

NUM_AR_ITERS = 50
TOTAL_ITERS = 150

VIDEO_EXTENSION_OVERRIDE = None
PCA_MODE = "load" # 8-keypoint PCA is pre-fitted in amph_project_8kp/pca.p

INIT_KAPPA = 150000
DECREASE_KAPPA_FACTOR = 10
MIXED_MAP_ITERS = 1

DOWNSAMPLE = 2
NEW_FPS = 40

USE_BATCHED_INIT = True
USE_MEMORY_PATCH = True
FLIPPING_MODE = None
SUBSET_METHOD = "spread"
SUBSET_FRACTION = 1.0

SESSIONS_PKL = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_dirty.pkl"
MIN_RAW_FRAMES = 50_000


def get_sessions():
    import h5py
    sessions = pd.read_pickle(SESSIONS_PKL)
    if not sessions:
        raise ValueError(f"No sessions in {SESSIONS_PKL}")

    filtered = []
    skipped = 0
    for s in sessions:
        h5_path = None
        if hasattr(s, 'tracking_refs_df') and s.tracking_refs_df is not None and not s.tracking_refs_df.empty:
            h5_path = s.tracking_refs_df['path'].iloc[0]
        if not h5_path or not os.path.exists(h5_path):
            skipped += 1
            continue
        try:
            with h5py.File(h5_path, 'r') as f:
                n = f['df_with_missing/table'].shape[0]
            if n >= MIN_RAW_FRAMES:
                filtered.append(s)
            else:
                print(f"[filter] Skipping {s.session_id}: only {n} raw frames (< {MIN_RAW_FRAMES})")
                skipped += 1
        except Exception:
            skipped += 1

    print(f"[filter] Kept {len(filtered)} sessions, skipped {skipped} (missing/short).")
    return filtered


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "amph_8kp_k150k_run_01"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    save_every_n_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    print("=" * 80)
    print("  KEYPOINT-MOSEQ 8-KEYPOINT PRODUCTION RUN")
    print(f"  Model Name:         {model_name}")
    print(f"  Seed:               {seed}")
    print(f"  Project Dir:        {PROJECT_DIR}")
    print(f"  Keypoints:          8 (Informative Core)")
    print(f"  Latent Dim:         {LATENT_DIM} (95.52% EVR)")
    print(f"  AR / SLDS Iters:    {NUM_AR_ITERS} / {TOTAL_ITERS} (Total: {NUM_AR_ITERS + TOTAL_ITERS})")
    print(f"  Init Kappa:         {INIT_KAPPA} (SLDS Kappa: {INIT_KAPPA // DECREASE_KAPPA_FACTOR})")
    print(f"  Checkpoint Every:   {save_every_n_iters} iters")
    print(f"  Memory Patch:       {USE_MEMORY_PATCH} (Banded Cholesky + Parallel CPU FFBS)")
    print("=" * 80, flush=True)

    sessions = get_sessions()
    print(f"Loaded {len(sessions)} sessions.")

    kwargs = {
        "project_dir": PROJECT_DIR,
        "detector": DETECTOR,
        "cam": CAM,
        "subroutine": SUBROUTINE,
        "latent_dim": LATENT_DIM,
        "num_ar_iters": NUM_AR_ITERS,
        "total_iters": TOTAL_ITERS,
        "override_video_extension": VIDEO_EXTENSION_OVERRIDE,
        "pca_mode": PCA_MODE,
        "model_name": model_name,
        "init_kappa": INIT_KAPPA,
        "decrease_kappa_factor": DECREASE_KAPPA_FACTOR,
        "mixed_map_iters": MIXED_MAP_ITERS,
        "seed": seed,
        "downsample_rate": DOWNSAMPLE,
        "new_fps": NEW_FPS,
        "use_batched_init": USE_BATCHED_INIT,
        "use_memory_patch": USE_MEMORY_PATCH,
        "flipping_mode": FLIPPING_MODE,
        "subset_method": SUBSET_METHOD,
        "subset_fraction": SUBSET_FRACTION,
        "save_every_n_iters": save_every_n_iters,
    }

    t0 = time.time()
    results = run_pipeline(sessions, **kwargs)
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print(f"✓ MODEL {model_name} COMPLETE in {elapsed/60:.2f} min ({elapsed:.1f}s)")
    print(f"  Median syllable duration: {results.get('median_duration_frames', 'N/A')} frames")
    print("=" * 80)

    # Save summary metadata
    summary_path = os.path.join(PROJECT_DIR, model_name, "model_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_data = {
        "model_name": model_name,
        "seed": seed,
        "latent_dim": LATENT_DIM,
        "keypoint_count": 8,
        "num_ar_iters": NUM_AR_ITERS,
        "total_iters": TOTAL_ITERS,
        "iters_done": NUM_AR_ITERS + TOTAL_ITERS,
        "init_kappa": INIT_KAPPA,
        "slds_kappa": INIT_KAPPA // DECREASE_KAPPA_FACTOR,
        "median_duration_frames": results.get("median_duration_frames"),
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
