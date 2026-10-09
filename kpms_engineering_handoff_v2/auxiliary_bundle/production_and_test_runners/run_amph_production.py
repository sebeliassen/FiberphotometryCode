#!/usr/bin/env python3
"""
run_amph_production.py

Production runner for Keypoint-MoSeq models on the Jonathan AMPH dataset.
Configuration:
  - 50 AR iterations + 150 SLDS iterations (200 total)
  - Init Kappa = 150,000 (SLDS Kappa = 15,000 via factor of 10)
  - Checkpointing every 50 iterations (conserves disk space)
  - All memory & speed optimizations active (Banded Cholesky + CPU HMM + GPU emissions)
  - Downsampled 2x -> 40 FPS

Usage:
  python run_amph_production.py [MODEL_NAME] [SEED] [SAVE_EVERY_N_ITERS]
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

# Add repo root to path for unpickling artifacts that reference 'syllable_generation'
sys.path.insert(0, "/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
np.bool8 = np.bool_

import json
import time
from pathlib import Path
from kpms_pipeline import run_pipeline
import pandas as pd

# ========= AMPH CONFIG =========
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "amph_project")
DETECTOR = "deeplabcut"
CAM = "center_cam"
SUBROUTINE = "jonathan"
LATENT_DIM = 10

NUM_AR_ITERS = 50
TOTAL_ITERS = 150

VIDEO_EXTENSION_OVERRIDE = None
PCA_MODE = "load_or_fit"

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


# ========= sessions provider =========
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


# ========= tiny io helper =========
class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32, np.float16)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)

def _write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NpEncoder)


# =====================================
if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "amph_prod_k150k_run_01"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    save_every_n_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    print("=" * 75)
    print(f"  AMPH PRODUCTION RUN: {model_name} (Seed: {seed})")
    print("=" * 75)
    print(f"Project dir:        {PROJECT_DIR}")
    print(f"Model name:         {model_name}")
    print(f"Seed:               {seed}")
    print(f"AR iters:           {NUM_AR_ITERS}")
    print(f"SLDS iters:         {TOTAL_ITERS} (Total: {NUM_AR_ITERS + TOTAL_ITERS})")
    print(f"Init kappa:         {INIT_KAPPA}")
    print(f"SLDS kappa:         {INIT_KAPPA / DECREASE_KAPPA_FACTOR}")
    print(f"Checkpoint every:   {save_every_n_iters} iters (Disk safe)")
    print(f"Downsample:         x{DOWNSAMPLE} (FPS -> {NEW_FPS})")
    print(f"Memory patch:       {USE_MEMORY_PATCH}")
    print(f"Batched init:       {USE_BATCHED_INIT}")
    print("=" * 75)
    print()

    sessions = get_sessions()
    print(f"Total sessions: {len(sessions)}")

    pipeline_kwargs = {
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

    wall_start = time.time()
    results = run_pipeline(sessions, **pipeline_kwargs)
    wall_end = time.time()

    wall_total = wall_end - wall_start
    total_iters_done = NUM_AR_ITERS + TOTAL_ITERS

    print()
    print("=" * 75)
    print(f"  PRODUCTION RUN COMPLETE: {model_name}")
    print(f"  Total wall-clock time:  {wall_total:.2f}s ({wall_total/60:.1f} min)")
    print(f"  Iterations completed:   {total_iters_done}")
    print(f"  Avg time per iteration: {wall_total/total_iters_done:.2f}s")
    print(f"  Frames used:            {results.get('frames_used', 'N/A')}")
    print(f"  Median syllable dur:    {results.get('median_duration_frames', 'N/A')} frames")
    print("=" * 75)

    # Save summary
    summary = {
        **pipeline_kwargs,
        **results,
        "wall_clock_seconds": wall_total,
        "avg_seconds_per_iter": wall_total / total_iters_done,
        "save_every_n_iters": save_every_n_iters,
    }
    model_dir = os.path.join(PROJECT_DIR, summary["model_name"])
    model_summary_path = os.path.join(model_dir, "model_summary.json")
    _write_json(model_summary_path, summary)
    print(f"Saved model summary to: {model_summary_path}")
