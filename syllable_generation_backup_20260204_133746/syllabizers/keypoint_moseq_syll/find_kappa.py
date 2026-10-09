#this is a module for binary kappa search

import math
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from kpms_pipeline import run_pipeline

import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

# ========= constants (edit these) =========
PROJECT_DIR = "amph_project"
DETECTOR = "deeplabcut"
CAM = "center_cam"
SUBROUTINE = "jonathan"

# --- NEW CONSTANTS ---
DOWNSAMPLE = 2
NEW_FPS = 40
# ---------------------

TARGET_FRAMES = 16
TOL_FRAMES = 0
MAX_STEPS = 12
STEP_FRACTION = 0.35
START_KAPPA = 1e7
KAPPA_BOUNDS = (1e2, 1e10)

NUM_AR_ITERS = 25
NUM_SLDS_ITERS = 50
DECREASE_KAPPA_FACTOR = 10

SUBSET_FRACTION = 0.1
SUBSET_BLOCKS = 10
SUBSET_BLOCK_LEN = None
SUBSET_SEED = 0

PCA_MODE = "load"  # full runs still load; subset runs compute temp PCA in-memory
MODEL_NAME_PREFIX = "kappa-scout"
RUN_TAG = "amph_sweep"

BASE_MIXED_MAP_ITERS = 24

VIDEO_EXTENSION_OVERRIDE = None

SESSIONS_PKL = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions.pkl"

def get_sessions():
    sessions = pd.read_pickle(SESSIONS_PKL)
    if not sessions:
        raise ValueError(f"No sessions in {SESSIONS_PKL}")
    return sessions


def _next_kappa(log_lo, log_hi, step_fraction, toward_hi=True):
    if toward_hi:
        return log_lo + step_fraction * (log_hi - log_lo)
    else:
        return log_hi - step_fraction * (log_hi - log_lo)


def _write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    assert 0 < STEP_FRACTION <= 0.5, "STEP_FRACTION should be in (0, 0.5]."
    sessions = get_sessions()

    log_lo = math.log(max(KAPPA_BOUNDS[0], 1e-12))
    log_hi = math.log(min(KAPPA_BOUNDS[1], 1e12))
    log_k = math.log(START_KAPPA)

    history = []  # list of dicts: {kappa, median_frames, model_name, frames_used, model_dir}

    for step in range(1, MAX_STEPS + 1):
        kappa_val = float(math.exp(log_k))
        model_name = f"{MODEL_NAME_PREFIX}-{(RUN_TAG+'-') if RUN_TAG else ''}{step:02d}-k{kappa_val:.3e}"

        start_dt = datetime.now().astimezone()
        summary = run_pipeline(
            sessions,
            project_dir=PROJECT_DIR,
            detector=DETECTOR,
            cam=CAM,
            subroutine=SUBROUTINE,
            latent_dim=10,
            num_ar_iters=NUM_AR_ITERS,
            total_iters=NUM_SLDS_ITERS,
            override_video_extension=VIDEO_EXTENSION_OVERRIDE,
            pca_mode=PCA_MODE,
            subset_fraction=SUBSET_FRACTION,
            subset_blocks=SUBSET_BLOCKS,
            subset_block_len=SUBSET_BLOCK_LEN,
            subset_seed=SUBSET_SEED,
            model_name=model_name,
            init_kappa=kappa_val,
            decrease_kappa_factor=DECREASE_KAPPA_FACTOR,
            mixed_map_iters=BASE_MIXED_MAP_ITERS,
            downsample_rate=DOWNSAMPLE,
            new_fps=NEW_FPS
        )

        median = float(summary["median_duration_frames"])
        frames_used = int(summary["frames_used"])
        model_name_used = summary["model_name"]  # in case KPMS normalized the name
        model_dir = os.path.join(PROJECT_DIR, model_name_used)

        finish_dt = datetime.now().astimezone()
        finished_at = finish_dt.isoformat(timespec="seconds")
        started_at  = start_dt.isoformat(timespec="seconds")
        duration_s  = (finish_dt - start_dt).total_seconds()

        # Per-model probe JSON (saved where the model is saved)
        probe_json = {
            "project_dir": PROJECT_DIR,
            "model_name": model_name_used,
            "model_dir": model_dir,
            
            "kappa_init": summary.get("kappa_init"),
            "kappa_slds": summary.get("kappa_slds"),
            
            "median_duration_frames": median,
            "frames_used": frames_used,
            "target_frames": TARGET_FRAMES,
            "tol_frames": TOL_FRAMES,
            
            "num_ar_iters": NUM_AR_ITERS,
            "num_slds_iters": NUM_SLDS_ITERS,
            "decrease_kappa_factor": DECREASE_KAPPA_FACTOR,
            
            "subset_fraction": SUBSET_FRACTION,
            "subset_blocks": SUBSET_BLOCKS,
            "subset_block_len": SUBSET_BLOCK_LEN,
            "subset_seed": SUBSET_SEED,
            
            "detector": summary.get("detector"),
            "cam": summary.get("cam"),
            "latent_dim": summary.get("latent_dim"),
            "video_dir": summary.get("video_dir"),
            "video_extension": summary.get("video_extension"),
            "pca_mode": summary.get("pca_mode"),
            "subroutine": summary.get("subroutine"),

            "started_at": started_at,      
            "finished_at": finished_at,    
            "duration_seconds": duration_s,
        }
        _write_json(os.path.join(model_dir, "kappa_probe.json"), probe_json)

        history.append({
            "kappa": kappa_val,
            "median_frames": median,
            "model_name": model_name_used,
            "frames_used": frames_used,
            "model_dir": model_dir,
            "started_at": started_at,      
            "finished_at": finished_at,    
            "duration_seconds": duration_s,
        })


        print(f"[step {step}/{MAX_STEPS}] "
              f"kappa={kappa_val:.3e} → median={median:.2f} frames "
              f"(target={TARGET_FRAMES}, tol={TOL_FRAMES}, frames_used={frames_used})")

        # Convergence check
        if np.isfinite(median) and abs(median - TARGET_FRAMES) <= TOL_FRAMES:
            print("\nConverged within tolerance.")
            break

        # Direction
        if not np.isfinite(median):
            log_k = min(log_k + 0.5, log_hi)  # nudge up if degenerate
            continue

        if median < TARGET_FRAMES:
            log_lo = max(log_lo, log_k)  # too short → increase kappa
            log_k = _next_kappa(log_lo, log_hi, STEP_FRACTION, toward_hi=True)
        else:
            log_hi = min(log_hi, log_k)  # too long → decrease kappa
            log_k = _next_kappa(log_lo, log_hi, STEP_FRACTION, toward_hi=False)

        log_k = max(min(log_k, log_hi), log_lo)

    # Pick best-so-far
    def _score(h):  # lower is better
        m = h["median_frames"]
        return abs(m - TARGET_FRAMES) if np.isfinite(m) else float("inf")

    best = min(history, key=_score) if history else None

    if best:
        # Save whole scan history into the BEST model’s folder
        scan_json = {
            "project_dir": PROJECT_DIR,
            "target_frames": TARGET_FRAMES,
            "tol_frames": TOL_FRAMES,
            "max_steps": MAX_STEPS,
            "step_fraction": STEP_FRACTION,
            "start_kappa": START_KAPPA,
            "bounds": list(KAPPA_BOUNDS),
            "num_ar_iters": NUM_AR_ITERS,
            "num_slds_iters": NUM_SLDS_ITERS,
            "decrease_kappa_factor": DECREASE_KAPPA_FACTOR,
            "subset_fraction": SUBSET_FRACTION,
            "subset_blocks": SUBSET_BLOCKS,
            "subset_block_len": SUBSET_BLOCK_LEN,
            "subset_seed": SUBSET_SEED,
            "history": history,
            "best": best,
        }
        _write_json(os.path.join(best["model_dir"], "kappa_scan_history.json"), scan_json)

        print(f"\nChosen kappa ≈ {best['kappa']:.3e} "
              f"(median={best['median_frames']:.2f} frames) "
              f"from model '{best['model_name']}'")
        print(f"History saved to {os.path.join(best['model_dir'], 'kappa_scan_history.json')}")
    else:
        print("\nNo runs executed? Check your sessions loader.")


if __name__ == "__main__":
    main()