#!/usr/bin/env python3
"""
probe_specific_kappas.py
------------------------
Sequentially probes specific kappa candidates:
[3.0e12, 1.0e13, 3.0e13]
on the 23-keypoint multi-view 60 FPS cohort snippets.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.18"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
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

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXPERIMENT_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/multiview_kappa_search"
PROJECT_DIR = str(EXPERIMENT_DIR / "project")
H5_DIR = EXPERIMENT_DIR / "fused_h5"
NTFY_URL = "https://ntfy.sh/woody_videos_seb"

PROBE_KAPPAS = [3.0e12, 1.0e13, 3.0e13]
TARGET_FRAMES = 24.0
NUM_AR_ITERS = 15
NUM_SLDS_ITERS = 25
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
    sub_coords = {}
    sub_confs = {}
    for name, c_arr in coords.items():
        total_len = len(c_arr)
        target_len = int(total_len * frac)
        block_len = max(50, target_len // n_blocks)
        margin = total_len // (n_blocks + 1)
        for b_idx in range(n_blocks):
            center = (b_idx + 1) * margin
            b_start = max(0, center - block_len // 2)
            b_end = min(total_len, b_start + block_len)
            sub_key = f"{name}_block{b_idx}"
            sub_coords[sub_key] = c_arr[b_start:b_end]
            sub_confs[sub_key] = confs[name][b_start:b_end]
    return sub_coords, sub_confs

def main():
    print("=" * 60)
    print("PROBING SPECIFIC KAPPA CANDIDATES")
    print(f"Candidates: {PROBE_KAPPAS}")
    print(f"Target syllable dwell: {TARGET_FRAMES:.1f} frames (400ms @ 60 FPS)")
    print("=" * 60)

    send_ntfy(
        "MoSeq Specific Kappa Probe Started",
        f"Probing candidates: {PROBE_KAPPAS}\nTarget: {TARGET_FRAMES} frames (60 FPS)"
    )

    # 1. Load Fused Keypoints
    coordinates, confidences, bodyparts = kpms.load_keypoints(str(H5_DIR), format="deeplabcut")
    sub_coords, sub_confs = create_snippet_subset(coordinates, confidences, n_blocks=N_BLOCKS, frac=FRAC_PER_SESSION)

    # 2. Format Data & Load PCA
    cfg = kpms.load_config(PROJECT_DIR)
    data, metadata = kpms.format_data(sub_coords, sub_confs, **cfg)
    from jax_moseq.utils.debugging import convert_data_precision
    data = convert_data_precision(data, np.float64)
    data["Y"] = np.asarray(data["Y"])
    pca = kpms.load_pca(PROJECT_DIR)

    results = []
    for idx, kappa_val in enumerate(PROBE_KAPPAS, 1):
        print(f"\n--- PROBE {idx}/{len(PROBE_KAPPAS)}: Testing kappa = {kappa_val:.3e} ---")

        step_cfg = dict(cfg)
        step_cfg["trans_hypparams"] = dict(step_cfg["trans_hypparams"])
        step_cfg["trans_hypparams"]["kappa"] = kappa_val
        kpms.update_config(PROJECT_DIR, **step_cfg)

        rng_key = jax.random.PRNGKey(100 + idx)
        model = kpms.init_model(data, pca=pca, seed=rng_key, **step_cfg)

        start_time = datetime.now()
        model, _ = kpms.fit_model(
            model, data, metadata,
            project_dir=PROJECT_DIR,
            model_name=f"probe_val_{idx}",
            ar_only=True,
            start_iter=0,
            num_iters=NUM_AR_ITERS,
            parallel_message_passing=False,
            save_every_n_iters=None,
            generate_progress_plots=False,
        )

        model, _ = kpms.fit_model(
            model, data, metadata,
            project_dir=PROJECT_DIR,
            model_name=f"probe_val_{idx}",
            ar_only=False,
            start_iter=NUM_AR_ITERS,
            num_iters=NUM_AR_ITERS + NUM_SLDS_ITERS,
            parallel_message_passing=False,
            save_every_n_iters=None,
            generate_progress_plots=False,
        )
        elapsed_s = (datetime.now() - start_time).total_seconds()

        z = np.array(model["states"]["z"])
        mask = np.array(data["mask"])
        durations = get_durations(z, mask)
        median_frames = float(np.median(durations)) if len(durations) > 0 else float("nan")
        median_ms = (median_frames / 60.0) * 1000.0

        print(f"Probe {idx} finished in {elapsed_s:.1f}s:")
        print(f"  kappa = {kappa_val:.3e} -> median dwell = {median_frames:.2f} frames ({median_ms:.1f} ms) [target: {TARGET_FRAMES:.1f} f]")

        res = {
            "probe": idx,
            "kappa": kappa_val,
            "median_frames": median_frames,
            "median_ms": median_ms,
            "elapsed_seconds": elapsed_s
        }
        results.append(res)

        msg = (
            f"Probe {idx}/{len(PROBE_KAPPAS)}: kappa={kappa_val:.2e}\n"
            f"Median dwell: {median_frames:.2f} frames ({median_ms:.1f} ms)\n"
            f"Target: {TARGET_FRAMES:.1f} frames (400 ms @ 60 FPS)"
        )
        send_ntfy(f"MoSeq Probe {idx}/{len(PROBE_KAPPAS)}", msg)

        # If we hit 24.0 frames or overshot to 25+, we know the behavior
        if median_frames >= 24.0:
            print(f"Reached or exceeded target ({median_frames:.2f} >= {TARGET_FRAMES:.1f})!")
            if idx == 1 and median_frames == 24.0:
                print("Hit 24.0 frames exactly! Continuing to probe 1e13 to map the sensitivity curve...")

    # Save probe results
    out_json = EXPERIMENT_DIR / "probe_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved probe results to {out_json}")

    send_ntfy("MoSeq Specific Probes Completed!", f"Results saved to {out_json.name}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err = f"Probe script failed: {e}\n\n{traceback.format_exc()}"
        print(err)
        send_ntfy("MoSeq Probe ERROR", err)
        sys.exit(1)
