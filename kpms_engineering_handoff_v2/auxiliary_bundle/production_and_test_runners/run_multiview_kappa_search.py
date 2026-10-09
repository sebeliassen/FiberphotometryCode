#!/usr/bin/env python3
"""
run_multiview_kappa_search.py
-----------------------------
Runs unconstrained Kappa search (kappa in [10^2, 10^10]) on the 23-keypoint
multi-view cohort using 5 uniform snippet blocks per session (15% total data).
Sends live notifications to ntfy.sh/woody_videos_seb.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.18"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import math
import sys
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import jax
import keypoint_moseq as kpms
from keypoint_moseq.viz import get_durations

# --- Configuration ---
REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXPERIMENT_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/multiview_kappa_search"
PROJECT_DIR = str(EXPERIMENT_DIR / "project")
H5_DIR = EXPERIMENT_DIR / "fused_h5"

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

# Search Parameters
KAPPA_BOUNDS = (1.0e10, 5.0e12)
START_KAPPA = 1.0e10
TARGET_FRAMES = 24.0  # 400ms at 60 FPS
TOL_FRAMES = 1.0  # +/- 1.0 frame (~16.7ms)
MAX_STEPS = 10
STEP_FRACTION = 0.35

NUM_AR_ITERS = 15
NUM_SLDS_ITERS = 25

# Subsetting: 5 blocks across each session, totaling 15%
N_BLOCKS = 5
FRAC_PER_SESSION = 0.15

def send_ntfy(title: str, message: str):
    try:
        req = urllib.request.Request(
            NTFY_URL,
            data=message.encode("utf-8"),
            headers={"Title": title}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass
    except Exception as e:
        print(f"[warning] Failed to send ntfy notification: {e}")

def create_snippet_subset(coords: dict, confs: dict, n_blocks: int = 5, frac: float = 0.15):
    """
    Extracts n_blocks evenly spaced snippets across each session sequence,
    preserving continuous temporal blocks for AR-SLDS modeling.
    """
    sub_coords = {}
    sub_confs = {}

    for name, c_arr in coords.items():
        total_len = len(c_arr)
        target_len = int(total_len * frac)
        block_len = max(50, target_len // n_blocks)

        # Determine start indices for n_blocks spread across the session
        # e.g. at 10%, 30%, 50%, 70%, 90%
        margin = total_len // (n_blocks + 1)
        for b_idx in range(n_blocks):
            center = (b_idx + 1) * margin
            b_start = max(0, center - block_len // 2)
            b_end = min(total_len, b_start + block_len)

            sub_key = f"{name}_block{b_idx}"
            sub_coords[sub_key] = c_arr[b_start:b_end]
            sub_confs[sub_key] = confs[name][b_start:b_end]

    return sub_coords, sub_confs

def _next_kappa(log_lo, log_hi, step_fraction, toward_hi=True):
    if toward_hi:
        return log_lo + step_fraction * (log_hi - log_lo)
    else:
        return log_hi - step_fraction * (log_hi - log_lo)

def main():
    print("=" * 60)
    print("STARTING 23-KEYPOINT MULTI-VIEW KAPPA SEARCH")
    print(f"Bounds: [{KAPPA_BOUNDS[0]:.1e}, {KAPPA_BOUNDS[1]:.1e}] | Start: {START_KAPPA:.1e}")
    print(f"Target syllable dwell: {TARGET_FRAMES:.1f} frames (400ms @ 30 FPS)")
    print("=" * 60)

    send_ntfy(
        "MoSeq Multi-View Kappa Search Started",
        f"Multi-view (23 keypoints) kappa scan launched!\n"
        f"Bounds: [{KAPPA_BOUNDS[0]:.1e}, {KAPPA_BOUNDS[1]:.1e}], start: {START_KAPPA:.1e}\n"
        f"Target: {TARGET_FRAMES} frames (60 FPS)\n"
        f"Sessions: 8 (4 Monomer, 4 PFF across 16w & 24w)"
    )

    # 1. Load Fused Keypoints
    h5_files = sorted(list(H5_DIR.glob("*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No fused H5 files found in {H5_DIR}")
    print(f"Found {len(h5_files)} fused H5 files:")
    for f in h5_files:
        print(f"  {f.name}")

    coordinates, confidences, bodyparts = kpms.load_keypoints(str(H5_DIR), format="deeplabcut")
    print(f"\nLoaded {len(coordinates)} sessions with {len(bodyparts)} bodyparts.")

    # 2. Extract 5 Snippet Blocks (15% per session)
    print(f"Extracting {N_BLOCKS} snippet blocks per session ({FRAC_PER_SESSION*100:.0f}% total)...")
    sub_coords, sub_confs = create_snippet_subset(coordinates, confidences, n_blocks=N_BLOCKS, frac=FRAC_PER_SESSION)
    total_sub_frames = sum(len(v) for v in sub_coords.values())
    print(f"Subset contains {len(sub_coords)} snippet sequences totaling {total_sub_frames} frames.")

    # 3. Format Data & PCA
    cfg = kpms.load_config(PROJECT_DIR)
    data, metadata = kpms.format_data(sub_coords, sub_confs, **cfg)
    from jax_moseq.utils.debugging import convert_data_precision
    data = convert_data_precision(data, np.float64)
    data["Y"] = np.asarray(data["Y"])

    pca_file = Path(PROJECT_DIR) / "pca.asdf"
    if pca_file.exists():
        print(f"Loading existing PCA from {pca_file.name}...")
        pca = kpms.load_pca(PROJECT_DIR)
    else:
        print("Fitting fresh 23-keypoint multi-view PCA...")
        pca = kpms.fit_pca(**data, **cfg)
        kpms.save_pca(pca, PROJECT_DIR)
        print("Saved fresh PCA.")

    # 4. Search Loop
    log_lo = math.log(max(KAPPA_BOUNDS[0], 1e-12))
    log_hi = math.log(KAPPA_BOUNDS[1])
    log_k  = math.log(START_KAPPA)

    history = []

    for step in range(1, MAX_STEPS + 1):
        kappa_val = float(math.exp(log_k))
        print(f"\n--- STEP {step}/{MAX_STEPS}: Testing kappa = {kappa_val:.3e} ---")

        step_cfg = dict(cfg)
        step_cfg["trans_hypparams"] = dict(step_cfg["trans_hypparams"])
        step_cfg["trans_hypparams"]["kappa"] = kappa_val
        kpms.update_config(PROJECT_DIR, **step_cfg)

        # Initialize model
        rng_key = jax.random.PRNGKey(step)
        model = kpms.init_model(data, pca=pca, seed=rng_key, **step_cfg)

        # Fit AR iterations
        start_time = datetime.now()
        model, model_name_used = kpms.fit_model(
            model, data, metadata,
            project_dir=PROJECT_DIR,
            model_name=f"probe_step_{step}",
            ar_only=True,
            start_iter=0,
            num_iters=NUM_AR_ITERS,
            parallel_message_passing=False,
            save_every_n_iters=None,
            generate_progress_plots=False,
        )

        # Run SLDS iterations
        model, model_name_used = kpms.fit_model(
            model, data, metadata,
            project_dir=PROJECT_DIR,
            model_name=f"probe_step_{step}",
            ar_only=False,
            start_iter=NUM_AR_ITERS,
            num_iters=NUM_AR_ITERS + NUM_SLDS_ITERS,
            parallel_message_passing=False,
            save_every_n_iters=None,
            generate_progress_plots=False,
        )
        elapsed_s = (datetime.now() - start_time).total_seconds()

        # Evaluate Syllable Dwell Times
        z = np.array(model["states"]["z"])
        mask = np.array(data["mask"])
        durations = get_durations(z, mask)
        median_frames = float(np.median(durations)) if len(durations) > 0 else float("nan")
        median_ms = (median_frames / 60.0) * 1000.0

        print(f"Step {step} finished in {elapsed_s:.1f}s:")
        print(f"  kappa = {kappa_val:.3e} -> median dwell = {median_frames:.2f} frames ({median_ms:.1f} ms) [target: {TARGET_FRAMES:.1f} f]")

        history.append({
            "step": step,
            "kappa": kappa_val,
            "median_frames": median_frames,
            "median_ms": median_ms,
            "elapsed_seconds": elapsed_s,
        })

        # Live ntfy notification
        step_msg = (
            f"Step {step}/{MAX_STEPS}: kappa={kappa_val:.2e}\n"
            f"Median dwell: {median_frames:.2f} frames ({median_ms:.1f} ms)\n"
            f"Target: {TARGET_FRAMES:.1f} frames (~400 ms @ 60 FPS)\n"
            f"Step time: {elapsed_s:.1f}s"
        )
        send_ntfy(f"MoSeq Multi-View Step {step}/{MAX_STEPS}", step_msg)

        # Check convergence
        if np.isfinite(median_frames) and abs(median_frames - TARGET_FRAMES) <= TOL_FRAMES:
            print(f"\nCONVERGED within tolerance ({abs(median_frames - TARGET_FRAMES):.2f} <= {TOL_FRAMES})!")
            break

        # Binary / interpolation search update
        if not np.isfinite(median_frames):
            log_k = min(log_k + 0.5, log_hi)
            continue

        if median_frames < TARGET_FRAMES:
            # Dwell time too short -> syllables switch too fast -> increase kappa to make transitions stickier
            log_lo = max(log_lo, log_k)
            log_k = _next_kappa(log_lo, log_hi, STEP_FRACTION, toward_hi=True)
            print(f"  Syllables too short ({median_frames:.2f} < {TARGET_FRAMES}) -> increasing kappa.")
        else:
            # Dwell time too long -> syllables are too sticky -> decrease kappa
            log_hi = min(log_hi, log_k)
            log_k = _next_kappa(log_lo, log_hi, STEP_FRACTION, toward_hi=False)
            print(f"  Syllables too long ({median_frames:.2f} > {TARGET_FRAMES}) -> decreasing kappa.")

        log_k = max(min(log_k, log_hi), log_lo)

    # 5. Summarize Results & Best Kappa
    def score_fn(h):
        m = h["median_frames"]
        return abs(m - TARGET_FRAMES) if np.isfinite(m) else float("inf")

    best_step = min(history, key=score_fn)
    best_kappa = best_step["kappa"]

    summary_data = {
        "target_frames": TARGET_FRAMES,
        "target_fps": 60,
        "bounds": [KAPPA_BOUNDS[0], KAPPA_BOUNDS[1]],
        "best_kappa": best_kappa,
        "best_median_frames": best_step["median_frames"],
        "best_median_ms": best_step["median_ms"],
        "history": history
    }

    summary_path = EXPERIMENT_DIR / "kappa_scan_history.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved scan history to {summary_path}")

    final_msg = (
        f"Multi-View Kappa Search Complete!\n"
        f"Optimal kappa* = {best_kappa:.3e}\n"
        f"Median duration = {best_step['median_frames']:.2f} frames ({best_step['median_ms']:.1f} ms)\n"
        f"Target was {TARGET_FRAMES:.1f} frames (400 ms @ 60 FPS)"
    )
    print("=" * 60)
    print(final_msg)
    print("=" * 60)
    send_ntfy("MoSeq Multi-View Kappa Search FINISHED!", final_msg)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err = f"Multi-View Kappa Search Failed: {e}\n\n{traceback.format_exc()}"
        print(err)
        send_ntfy("MoSeq Multi-View Kappa Search ERROR", err)
        sys.exit(1)
