#!/usr/bin/env python3
"""
run_8kp_scout.py
=============================================================================
Fast Scout Run for 8-Keypoint Candidate Skeleton:
- Latent Dim = 6 (Captures 95.52% of total pose variance)
- 25 AR iterations + 25 SLDS iterations (50 iters total)
- Test Init Kappa = 150,000 (SLDS Kappa = 15,000)
- Measures median syllable duration to confirm whether it hits the target ~15 frames.
=============================================================================
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import multiprocessing
import sys

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

import h5py
from kpms_pipeline import run_pipeline
from run_amph_production import get_sessions

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "amph_project_8kp")
MODEL_NAME = "scout_8kp_k150k"
LATENT_DIM = 6
NUM_AR_ITERS = 25
TOTAL_ITERS = 25

INIT_KAPPA = 150000
DECREASE_KAPPA_FACTOR = 10
MIXED_MAP_ITERS = 1
DOWNSAMPLE = 2
NEW_FPS = 40
SEED = 53

def send_ntfy(message, title="KPMS 8KP Scout Progress"):
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://ntfy.sh/woody_videos_seb",
            data=message.encode("utf-8"),
            headers={"Title": title}
        )
        with urllib.request.urlopen(req) as resp:
            pass
    except Exception as e:
        print(f"[ntfy notice] (ntfy quota notice: {e})")

def main():
    print("=" * 80)
    print("🚀 RUNNING FAST SCOUT RUN: 8-KEYPOINT SKELETON (25 AR + 25 SLDS)")
    print(f"   Project Dir: {PROJECT_DIR}")
    print(f"   Model Name:  {MODEL_NAME}")
    print(f"   Latent Dim:  {LATENT_DIM} (captures 95.52% EVR)")
    print(f"   Init Kappa:  {INIT_KAPPA} (SLDS Kappa: {INIT_KAPPA // DECREASE_KAPPA_FACTOR})")
    print(f"   Total Iters: {NUM_AR_ITERS} AR + {TOTAL_ITERS} SLDS = {NUM_AR_ITERS + TOTAL_ITERS} total")
    print("=" * 80, flush=True)

    send_ntfy(f"Starting 8-Keypoint Scout Run ({NUM_AR_ITERS} AR + {TOTAL_ITERS} SLDS, K={INIT_KAPPA}, Dim={LATENT_DIM}).")

    sessions = get_sessions()
    print(f"Loaded {len(sessions)} sessions.")

    pipeline_kwargs = {
        "project_dir": PROJECT_DIR,
        "detector": "deeplabcut",
        "cam": "center_cam",
        "subroutine": "jonathan",
        "latent_dim": LATENT_DIM,
        "num_ar_iters": NUM_AR_ITERS,
        "total_iters": TOTAL_ITERS,
        "override_video_extension": None,
        "pca_mode": "load", # Load the fitted 8kp PCA
        "model_name": MODEL_NAME,
        "init_kappa": INIT_KAPPA,
        "decrease_kappa_factor": DECREASE_KAPPA_FACTOR,
        "mixed_map_iters": MIXED_MAP_ITERS,
        "seed": SEED,
        "downsample_rate": DOWNSAMPLE,
        "new_fps": NEW_FPS,
        "use_batched_init": True,
        "use_memory_patch": True,
        "flipping_mode": None,
        "subset_method": "spread",
        "subset_fraction": 1.0,
        "save_every_n_iters": 25,
    }

    t0 = time.time()
    results = run_pipeline(sessions, **pipeline_kwargs)
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print(f"✓ SCOUT RUN COMPLETED in {elapsed/60:.2f} min ({elapsed:.1f}s)!")
    print("=" * 80, flush=True)

    # Inspect median syllable duration from results
    res_path = os.path.join(PROJECT_DIR, MODEL_NAME, "results.h5")
    if os.path.exists(res_path):
        all_durations = []
        with h5py.File(res_path, "r") as f:
            for s_name in f.keys():
                sylls = f[s_name]["syllable"][:]
                valid = sylls[sylls >= 0]
                if len(valid) == 0:
                    continue
                # RLE
                change_idx = np.where(valid[1:] != valid[:-1])[0] + 1
                starts = np.r_[0, change_idx]
                ends = np.r_[change_idx, len(valid)]
                durs = ends - starts
                all_durations.extend(durs)

        med_dur = float(np.median(all_durations))
        mean_dur = float(np.mean(all_durations))
        p25 = float(np.percentile(all_durations, 25))
        p75 = float(np.percentile(all_durations, 75))

        status_msg = (
            f"8-Keypoint Scout Completed ({elapsed/60:.1f}m):\n"
            f"Median Syllable Duration: {med_dur:.1f} frames (Target: ~15 frames / 375ms)\n"
            f"Mean: {mean_dur:.1f}, IQR: [{p25:.1f}, {p75:.1f}] frames."
        )
        print("\n" + status_msg, flush=True)
        send_ntfy(status_msg, title="KPMS 8KP Scout Results")

        # Save summary
        summary = {
            "model_name": MODEL_NAME,
            "latent_dim": LATENT_DIM,
            "init_kappa": INIT_KAPPA,
            "slds_kappa": INIT_KAPPA // DECREASE_KAPPA_FACTOR,
            "median_duration_frames": med_dur,
            "mean_duration_frames": mean_dur,
            "p25_duration_frames": p25,
            "p75_duration_frames": p75,
            "elapsed_seconds": elapsed,
            "hit_target_15": abs(med_dur - 15) <= 3
        }
        with open(os.path.join(PROJECT_DIR, MODEL_NAME, "scout_summary.json"), "w") as f_out:
            json.dump(summary, f_out, indent=2)

if __name__ == "__main__":
    main()
