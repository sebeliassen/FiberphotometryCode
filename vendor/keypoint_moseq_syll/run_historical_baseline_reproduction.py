#!/usr/bin/env python3
"""
run_historical_baseline_reproduction.py

Exact full-run reproduction of the December 19, 2025 baseline workload:
  - Original baseline (amph_full_run.log & pipeline_log.txt):
      * 50 AR iterations (took 48m 39s)
      * 201 SLDS iterations (took 17h 18m 46s @ 310.08s/iter)
      * Total iterations: 250 (iters_done = 250)
      * Seed: 42, Downsample: x2 (40 FPS), Init Kappa: 15000 (SLDS Kappa: 1500.0)
  - Full dataset: 30 sessions, 562 segmented sequences, 5,516,954 frames
  - Executed with the full monkeypatched optimization suite:
      * Pure Numba Banded Cholesky solver
      * Vectorized GPU segment_sum AR dynamics
      * GPU tensor-core emissions + Numba parallel CPU FFBS
      * Batched pose parameters & consolidated NaN checks
"""

import os
import sys

# Crucial numpy compatibility patch before importing bokeh/panel/kpms
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# GPU memory environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import multiprocessing

if multiprocessing.current_process().name != 'MainProcess':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, KPMS_DIR)

import json
import shutil
import time
import urllib.request
from datetime import datetime

import h5py
import pandas as pd
from kpms_pipeline import run_pipeline

PROJECT_DIR = os.path.join(KPMS_DIR, "amph_project")
MODEL_NAME = "amph_full_run_monkeypatched_reproduction"
OUTPUT_DIR = os.path.join(PROJECT_DIR, MODEL_NAME)
SESSIONS_PKL = os.path.join(REPO_ROOT, "syllable_generation/data/jonathan_amph_sessions_dirty.pkl")
LOG_DIR = os.path.join(KPMS_DIR, "logs")
MIN_RAW_FRAMES = 50_000
NTFY_URL = "https://ntfy.sh/woody_videos_seb"

# Historical reference constants
HISTORICAL_AR_TIME_SEC = 2919.0      # 48m 39s for 50 AR iters (57.25s/iter)
HISTORICAL_SLDS_TIME_SEC = 62326.0   # 17h 18m 46s for 201 SLDS iters (310.08s/iter)
HISTORICAL_TOTAL_SEC = 65245.0       # ~18 hours 7 minutes

from fpcode.common.notify import send_ntfy

def get_sessions():
    sessions = pd.read_pickle(SESSIONS_PKL)
    if not sessions:
        raise ValueError(f"No sessions found in {SESSIONS_PKL}")
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
                skipped += 1
        except Exception:
            skipped += 1
    print(f"[filter] Kept {len(filtered)} sessions, skipped {skipped} (missing/short).", flush=True)
    return filtered

def main():
    print("=" * 85, flush=True)
    print("  EXACT HISTORICAL BASELINE REPRODUCTION BENCHMARK: 50 AR + 200 SLDS", flush=True)
    print("=" * 85, flush=True)
    print(f"Timestamp:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Dataset:            {SESSIONS_PKL}", flush=True)
    print("Workload:           50 AR iters + 200 SLDS iters (Total: 250 iterations)", flush=True)
    print("Seed:               42 (Exact historical match)", flush=True)
    print("Downsample:         x2 (FPS -> 40)", flush=True)
    print("Init Kappa:         15000 (SLDS Kappa: 1500.0, matching pipeline_log.txt)", flush=True)
    print(f"Model Directory:    {OUTPUT_DIR}", flush=True)
    print("Historical Ref:     AR: 48m 39s | SLDS: 17h 18m 46s | Total: ~18.1 hours", flush=True)
    print("=" * 85, flush=True)

    if os.path.exists(OUTPUT_DIR):
        print(f"[clean] Removing previous output directory: {OUTPUT_DIR}", flush=True)
        shutil.rmtree(OUTPUT_DIR)

    sessions = get_sessions()
    print(f"[data] Loaded {len(sessions)} full experimental sessions.", flush=True)

    send_ntfy(
        "🚀 Started exact historical reproduction benchmark (50 AR + 200 SLDS = 250 iters) on full 5.5M frame cohort.\n"
        "Historical baseline: 18.1 hours (310 s/iter).\n"
        "Projected monkeypatched runtime: ~42-45 minutes.",
        title="KPMS Benchmark Launch"
    )

    pipeline_kwargs = {
        "project_dir": PROJECT_DIR,
        "detector": "deeplabcut",
        "cam": "center_cam",
        "subroutine": "jonathan",
        "latent_dim": 10,
        "num_ar_iters": 50,
        "total_iters": 200,   # 50 AR + 200 SLDS = 250 total iters
        "pca_mode": "load",
        "model_name": MODEL_NAME,
        "init_kappa": 15000,
        "decrease_kappa_factor": 10,
        "mixed_map_iters": 1,
        "seed": 42,
        "downsample_rate": 2,
        "new_fps": 40,
        "use_batched_init": True,
        "use_memory_patch": True,
        "subset_method": "spread",
        "subset_fraction": 1.0,
        "save_every_n_iters": 50,
    }

    t_start = time.perf_counter()
    results = run_pipeline(sessions, **pipeline_kwargs)
    t_total = time.perf_counter() - t_start

    # Compute Speedup Scorecard
    speedup_full = HISTORICAL_TOTAL_SEC / t_total

    print("\n" + "=" * 85, flush=True)
    print("                HISTORICAL REPRODUCTION SCORECARD", flush=True)
    print("=" * 85, flush=True)
    print(f"{'Metric':<32} | {'Dec 19, 2025 Baseline':<22} | {'Monkeypatched Engine':<22}", flush=True)
    print("-" * 85, flush=True)
    print(f"{'AR Phase (50 iters)':<32} | 48m 39s (57.25 s/iter)   | Measured in run log", flush=True)
    print(f"{'SLDS Phase (200 iters)':<32} | 17h 19m (310.08 s/iter)  | Measured in run log", flush=True)
    print(f"{'Total Wall-Clock Time':<32} | 18h 07m (65,245 s)       | {t_total:.1f} s ({t_total/60:.2f} min)", flush=True)
    print(f"{'Total Full-Run Speedup':<32} | 1.00x                    | {speedup_full:.2f}x Faster", flush=True)
    print(f"{'Active Syllables':<32} | Expected: 70-85          | {results.get('median_duration_frames', 'N/A')} frames median", flush=True)
    print(f"{'Output Directory':<32} | amph_full_run            | {MODEL_NAME}", flush=True)
    print("=" * 85, flush=True)

    # Save structured scorecard JSON
    scorecard_path = os.path.join(OUTPUT_DIR, "reproduction_scorecard.json")
    with open(scorecard_path, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "total_iters": 250,
            "ar_iters": 50,
            "slds_iters": 200,
            "seed": 42,
            "init_kappa": 15000,
            "slds_kappa": 1500.0,
            "historical_baseline_sec": HISTORICAL_TOTAL_SEC,
            "historical_slds_sec_per_iter": 310.08,
            "monkeypatched_total_sec": t_total,
            "speedup_factor": speedup_full,
            "frames_used": results.get("frames_used"),
            "median_duration_frames": results.get("median_duration_frames")
        }, f, indent=2)
    print(f"Scorecard saved to: {scorecard_path}", flush=True)

    # Send completion notification
    send_ntfy(
        f"✅ Full Historical Reproduction Complete!\n"
        f"Total Wall-Clock: {t_total/60:.1f} min ({t_total:.0f}s) for 250 iterations.\n"
        f"Historical Baseline: 18.1 hours (65,245s).\n"
        f"Measured Speedup: {speedup_full:.1f}x overall!",
        title="KPMS Benchmark Finished",
        tags="tada,checkered_flag,trophy"
    )

if __name__ == "__main__":
    main()
