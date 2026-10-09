#!/usr/bin/env python3
"""
train_full_multiview_model.py
-----------------------------
Trains the full 23-keypoint multi-view Keypoint-MoSeq model (50 AR + 150 SLDS iterations)
on the complete 8-session cohort (1,240,000 frames @ native 60.0 FPS).

Hyperparameters:
  - Init Kappa (Phase 1 AR): 6.0e12
  - SLDS Kappa (Phase 2 SLDS): 6.0e11 (10x reduction per protocol)
  - VRAM limit: <= 16 GB (safe for RTX 4090 alongside background DLC)
  - Batching: batched_init_model_by_video + apply_optimizations
"""

import os

# Cap XLA client allocation to ~16 GB (0.65 of 24.5 GB) to ensure <= 18 GB total GPU allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.65"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
from pathlib import Path

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll"))

import json
import time
import traceback
import urllib.request
from datetime import datetime

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import jax
import keypoint_moseq as kpms
from jax_moseq.utils.debugging import convert_data_precision
from keypoint_moseq.viz import get_durations

from syllable_generation.syllabizers.keypoint_moseq_syll.utils.batch_init import batched_init_model_by_video
from syllable_generation.syllabizers.keypoint_moseq_syll.utils.resample_batching import (
    apply_optimizations,
    set_batch_sizes,
)

# --- Configuration ---
EXPERIMENT_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/multiview_kappa_search"
PROJECT_DIR = str(EXPERIMENT_DIR / "project")
H5_DIR = EXPERIMENT_DIR / "fused_h5"
MODEL_NAME = "fullmodel_multiview_23kp_k7.5e13"
NTFY_URL = "https://ntfy.sh/woody_videos_seb"

INIT_KAPPA = 7.5e13
DECREASE_KAPPA_FACTOR = 10.0
SLDS_KAPPA = INIT_KAPPA / DECREASE_KAPPA_FACTOR  # 7.5e12

NUM_AR_ITERS = 50
NUM_SLDS_ITERS = 150
TOTAL_ITERS = NUM_AR_ITERS + NUM_SLDS_ITERS  # 200
SAVE_EVERY_N_ITERS = 25
TARGET_FPS = 60

def send_ntfy(title: str, message: str):
    try:
        clean_title = title.encode("ascii", "replace").decode("ascii")
        req = urllib.request.Request(
            NTFY_URL,
            data=message.encode("utf-8"),
            headers={"Title": clean_title}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass
    except Exception as e:
        print(f"[warning] Failed to send ntfy notification: {e}")

def main():
    print("=" * 70)
    print("STARTING FULL 23-KEYPOINT MULTI-VIEW MODEL TRAINING (60 FPS)")
    print(f"Model Name:      {MODEL_NAME}")
    print(f"Iterations:      {NUM_AR_ITERS} AR + {NUM_SLDS_ITERS} SLDS = {TOTAL_ITERS} total")
    print(f"Init Kappa (AR): {INIT_KAPPA:.2e}")
    print(f"SLDS Kappa:      {SLDS_KAPPA:.2e} (10x reduction)")
    print(f"Save Interval:   Every {SAVE_EVERY_N_ITERS} iterations")
    print("VRAM Cap:        ~16 GB (0.65 fraction)")
    print("=" * 70)

    send_ntfy(
        "KPMS Multi-View Full Model Started",
        f"Training {MODEL_NAME}\n"
        f"Schedule: 50 AR + 150 SLDS = 200 iters\n"
        f"Kappa: {INIT_KAPPA:.1e} (AR) -> {SLDS_KAPPA:.1e} (SLDS)\n"
        f"8 sessions, 1.24M frames @ 60 FPS"
    )

    # 1. Apply high-performance kernels with safe batch sizes
    apply_optimizations()
    set_batch_sizes(
        discrete_bs=8,
        continuous_chunk_size=25,
        scales_bs=30,
        heading_bs=15,
        gpu_loc_bs=30
    )

    # 2. Load 100% full dataset (8 sessions, 1.24M frames)
    h5_files = sorted(list(H5_DIR.glob("*.h5")))
    print(f"\nLoading {len(h5_files)} fused H5 sessions from {H5_DIR}...")
    coordinates, confidences, bodyparts = kpms.load_keypoints(str(H5_DIR), format="deeplabcut")
    total_frames = sum(len(v) for v in coordinates.values())
    print(f"Loaded {len(coordinates)} sessions totaling {total_frames:,} frames ({len(bodyparts)} bodyparts).")

    # 3. Format Data
    cfg = kpms.load_config(PROJECT_DIR)
    cfg["fps"] = TARGET_FPS
    cfg["trans_hypparams"]["kappa"] = INIT_KAPPA
    kpms.update_config(PROJECT_DIR, **cfg)

    print("\nFormatting 1.24M frames for modeling...")
    data, metadata = kpms.format_data(coordinates, confidences, **cfg)
    data = convert_data_precision(data, np.float64)
    data["Y"] = np.asarray(data["Y"])
    print(f"Formatted data array Y shape: {data['Y'].shape}")

    # Load validated PCA
    pca = kpms.load_pca(PROJECT_DIR)
    print("Loaded 23-keypoint multi-view PCA.")

    # 4. Check for existing checkpoint to support resumption
    model_save_dir = Path(PROJECT_DIR) / MODEL_NAME
    checkpoint_file = model_save_dir / "checkpoint.h5"
    current_iter = 0
    model = None

    if checkpoint_file.exists():
        try:
            print(f"\nFound existing checkpoint at {checkpoint_file}. Resuming...")
            model, data, metadata, current_iter = kpms.load_checkpoint(PROJECT_DIR, MODEL_NAME)
            data["Y"] = np.asarray(data["Y"])
            print(f"Resumed at iteration {current_iter}.")
        except Exception as e:
            print(f"Could not resume from checkpoint ({e}). Initializing fresh model...")
            model = None
            current_iter = 0

    if model is None:
        print("\nInitializing fresh model via batched initialization...")
        t_init = time.time()
        model = batched_init_model_by_video(
            data, pca, cfg, params_subset_size=5, batch_size=16, seed=jax.random.PRNGKey(42)
        )
        model = kpms.update_hypparams(model, kappa=float(INIT_KAPPA))
        current_iter = 0
        print(f"Model initialized in {time.time() - t_init:.1f}s.")

    start_wall_time = datetime.now()

    # 5. PHASE 1: AR-only iterations (0 -> 50)
    if current_iter < NUM_AR_ITERS:
        print("\n=======================================================")
        print(f"PHASE 1: AR-Only Fitting (iter {current_iter} -> {NUM_AR_ITERS})")
        print(f"Init Kappa: {INIT_KAPPA:.2e}")
        print("=======================================================")

        ar_step_start = time.time()
        model, _ = kpms.fit_model(
            model, data, metadata,
            project_dir=PROJECT_DIR,
            model_name=MODEL_NAME,
            ar_only=True,
            start_iter=current_iter,
            num_iters=NUM_AR_ITERS,
            parallel_message_passing=False,
            save_every_n_iters=SAVE_EVERY_N_ITERS,
            generate_progress_plots=True,
        )
        current_iter = NUM_AR_ITERS
        ar_duration_min = (time.time() - ar_step_start) / 60.0
        print(f"\n[Phase 1 Complete] 50 AR iterations finished in {ar_duration_min:.1f} min.")

        send_ntfy(
            "KPMS Phase 1 Complete (50 AR)",
            f"Phase 1 AR completed in {ar_duration_min:.1f} min.\n"
            f"Now transitioning to Phase 2 SLDS (10x kappa reduction: {SLDS_KAPPA:.1e})."
        )

    # 6. TRANSITION: 10x Kappa reduction for Phase 2 (SLDS)
    print("\n=======================================================")
    print("TRANSITION: Applying 10x Kappa Reduction for Phase 2 SLDS")
    print(f"  Old Kappa: {float(model['hypparams']['trans_hypparams']['kappa']):.2e}")
    model = kpms.update_hypparams(model, kappa=float(SLDS_KAPPA))
    print(f"  New Kappa: {float(model['hypparams']['trans_hypparams']['kappa']):.2e}")
    print("=======================================================")

    # 7. PHASE 2: SLDS iterations (50 -> 200 in chunks of 25)
    if current_iter < TOTAL_ITERS:
        print("\n=======================================================")
        print(f"PHASE 2: Full SLDS Fitting (iter {current_iter} -> {TOTAL_ITERS})")
        print(f"SLDS Kappa: {SLDS_KAPPA:.2e}")
        print("=======================================================")

        chunk_size = SAVE_EVERY_N_ITERS
        while current_iter < TOTAL_ITERS:
            next_target = min(current_iter + chunk_size, TOTAL_ITERS)
            chunk_start = time.time()
            print(f"\n>>> Running SLDS chunk: iter {current_iter} -> {next_target}...")

            model, _ = kpms.fit_model(
                model, data, metadata,
                project_dir=PROJECT_DIR,
                model_name=MODEL_NAME,
                ar_only=False,
                start_iter=current_iter,
                num_iters=next_target,
                parallel_message_passing=False,
                save_every_n_iters=SAVE_EVERY_N_ITERS,
                generate_progress_plots=True,
            )
            current_iter = next_target
            chunk_dur_min = (time.time() - chunk_start) / 60.0

            # Evaluate current median duration
            z = np.array(model["states"]["z"])
            mask = np.array(data["mask"])
            durations = get_durations(z, mask)
            med_frames = float(np.median(durations)) if len(durations) > 0 else float("nan")
            med_ms = (med_frames / TARGET_FPS) * 1000.0

            print(f"Checkpoint at iter {current_iter}/{TOTAL_ITERS} ({chunk_dur_min:.1f} min):")
            print(f"  Current median dwell: {med_frames:.2f} frames ({med_ms:.1f} ms)")

            msg = (
                f"Iter {current_iter}/{TOTAL_ITERS} finished ({chunk_dur_min:.1f} min):\n"
                f"Median dwell: {med_frames:.2f} frames ({med_ms:.1f} ms @ 60 FPS)\n"
                f"Kappa: {SLDS_KAPPA:.1e}"
            )
            send_ntfy(f"KPMS Iteration {current_iter}/{TOTAL_ITERS}", msg)

    total_elapsed_min = (datetime.now() - start_wall_time).total_seconds() / 60.0

    # 8. Final Evaluation & Summary
    print("\nExtracting final results and saving results.h5...")
    results = kpms.extract_results(model, metadata, PROJECT_DIR, MODEL_NAME, save_results=True)

    z_final = np.array(model["states"]["z"])
    mask_final = np.array(data["mask"]).astype(bool)
    valid_mask = mask_final[:, :z_final.shape[1]]
    durations = get_durations(z_final, mask_final)
    final_median_frames = float(np.median(durations))
    final_median_ms = (final_median_frames / TARGET_FPS) * 1000.0

    valid_z = z_final[valid_mask]
    unique_sylls, syll_counts = np.unique(valid_z, return_counts=True)
    num_active_sylls = len(unique_sylls)

    sort_idx = np.argsort(-syll_counts)
    top_sylls = [
        {"syllable": int(unique_sylls[i]), "count": int(syll_counts[i]), "usage_pct": float(syll_counts[i] / valid_mask.sum() * 100)}
        for i in sort_idx[:15]
    ]

    summary_data = {
        "model_name": MODEL_NAME,
        "total_iters": TOTAL_ITERS,
        "num_ar_iters": NUM_AR_ITERS,
        "num_slds_iters": NUM_SLDS_ITERS,
        "init_kappa": INIT_KAPPA,
        "decrease_kappa_factor": DECREASE_KAPPA_FACTOR,
        "slds_kappa": SLDS_KAPPA,
        "target_fps": TARGET_FPS,
        "total_frames_modeled": total_frames,
        "final_median_frames": final_median_frames,
        "final_median_ms": final_median_ms,
        "num_active_syllables": int(num_active_sylls),
        "top_15_syllables": top_sylls,
        "total_wall_time_minutes": total_elapsed_min,
        "finished_at": datetime.now().isoformat()
    }

    summary_path = model_save_dir / "model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved model summary to {summary_path}")

    final_msg = (
        f"Full Model Training SUCCESS!\n"
        f"Model: {MODEL_NAME}\n"
        f"Completed: 50 AR + 150 SLDS = 200 iters in {total_elapsed_min:.1f} min\n"
        f"Median dwell: {final_median_frames:.2f} frames ({final_median_ms:.1f} ms @ 60 FPS)\n"
        f"Active syllables: {num_active_sylls}\n"
        f"Saved to: {model_save_dir}"
    )
    print("=" * 70)
    print(final_msg)
    print("=" * 70)
    send_ntfy("KPMS Multi-View Training COMPLETE", final_msg)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err = f"Full Model Training Failed: {e}\n\n{traceback.format_exc()}"
        print(err)
        send_ntfy("❌ KPMS Multi-View Training FAILED", err)
        sys.exit(1)
