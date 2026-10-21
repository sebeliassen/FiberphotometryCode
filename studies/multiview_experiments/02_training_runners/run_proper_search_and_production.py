#!/usr/bin/env python3
"""
run_proper_search_and_production.py
===================================
1. Runs extended Kappa Scan specifically for `21kp_static`:
   - Search range: log10(kappa) in [15.0, 26.0] (1e15 to 1e26)
   - Target dwell: 24.0 frames (400ms @ 60fps) +/- 2.0 frames (22.0 to 26.0)
   - Bisection with 42% step fraction
   - Max 12 steps until strictly within tolerance
2. Updates kappa_scan_results.json with 21kp_static's true optimal scout kappa
   and production kappa (3x scout kappa).
3. Executes full 200-iteration production training across all 3 arms (17kp, 21kp_static, 21kp_dynamic)
   using isolated subprocesses to guarantee clean GPU memory.
4. Runs head-to-head evaluation and generates master scorecard.
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from itertools import groupby
from pathlib import Path

import numpy as np

from fpcode.common.paths import get_repo_root

# Unbuffered live stdout/stderr
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ["BROWSER"] = "/bin/true"
import webbrowser

webbrowser.get = lambda *args, **kwargs: None

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["USE_MEMORY_PATCH"] = "1"

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import matplotlib

matplotlib.use("Agg")

import keypoint_moseq as kpms

REPO_ROOT = Path(str(get_repo_root()))
sys.path.insert(0, str(REPO_ROOT))
EXP_DIR = REPO_ROOT / 'studies/multiview_experiments'
sys.path.insert(0, str(EXP_DIR))
KPMS_UTILS_DIR = REPO_ROOT / 'vendor/keypoint_moseq_syll'
sys.path.insert(0, str(KPMS_UTILS_DIR))

import utils.resample_batching as rb  # noqa: I001 -- Preserve historical runner import order.
from fpcode.kpms.checkpoints import apply_kpms_checkpoint_patch
from utils.batch_init import batched_init_model_by_video

PYTHON_BIN = sys.executable
RUNNER_SCRIPT = EXP_DIR / "02_training_runners/run_pilot_17kp_vs_21kp_comparison.py"
EVAL_SCRIPT = EXP_DIR / "03_evaluation_and_eml/evaluate_17kp_vs_21kp_comparison.py"
SCAN_JSON_PATH = EXP_DIR / "04_reports_and_figures/kappa_scan_results.json"

SCOUT_SIDS = [
    "2026-03-05T11_01_01_bottom_left",
    "2026-03-05T12_46_18_top_left"
]

TARGET_FRAMES = 24.0  # 400ms at 60 fps
TOL_FRAMES = 2.0     # 22.0 to 26.0 frames
STEP_FRACTION = 0.42 # conservative bisection


def compute_median_duration(z):
    durations = []
    if z.ndim == 1:
        z = z[None, :]
    for row in z:
        for val, grp in groupby(row):
            if val >= 0:
                durations.append(len(list(grp)))
    return float(np.median(durations)) if len(durations) > 0 else float("nan")


def slice_scout_data(full_data, full_meta, scout_sids):
    keys = full_meta[0]
    scout_mask = np.array([any(sid in str(k) for sid in scout_sids) for k in keys])
    scout_idx = np.where(scout_mask)[0]
    sub_data = {k: v[scout_idx] for k, v in full_data.items()}
    sub_meta = (keys[scout_idx], full_meta[1][scout_idx])
    return sub_data, sub_meta


def run_extended_scan_21kp_static():
    """
    Runs extended bisection scan specifically for 21kp_static starting at 1e15 up to 1e26.
    """
    arm = "21kp_static"
    arm_dir = EXP_DIR / f"pilot_comparison_{arm}"

    with open(arm_dir / "preprocessed_data.pkl", "rb") as f:
        full_data, full_meta = pickle.load(f)
    with open(arm_dir / "pca.p", "rb") as f:
        pca = pickle.load(f)
    config = kpms.load_config(str(arm_dir))

    sub_data, sub_meta = slice_scout_data(full_data, full_meta, SCOUT_SIDS)

    print("=" * 80)
    print(f"🎯 EXTENDED TARGETED KAPPA SCAN: {arm.upper()}")
    print(f"  Target Duration: {TARGET_FRAMES} frames (400ms @ 60fps) +/- {TOL_FRAMES} frames")
    print(f"  Acceptance Range: [{TARGET_FRAMES - TOL_FRAMES:.1f}, {TARGET_FRAMES + TOL_FRAMES:.1f}] frames")
    print("  Search Range:     [10^15.0, 10^26.0]")
    print(f"  Step Fraction:    {STEP_FRACTION:.2f}")
    print("=" * 80, flush=True)

    log_lo = 15.0
    log_hi = 26.0
    log_curr = 16.5 # Start slightly above 1e16 where it reached 19.0 frames
    max_steps = 12

    history = []
    best_step = None
    best_diff = float("inf")

    probe_base_dir = arm_dir / "kappa_probes"
    probe_base_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, max_steps + 1):
        kappa_val = float(10**log_curr)
        print(f"\n--- [ARM: {arm.upper()}] EXTENDED SCAN STEP {step}/{max_steps} ---")
        print(f"  Testing kappa: {kappa_val:.3e} (log10 = {log_curr:.3f})")
        print(f"  Current active bounds: [10^{log_lo:.3f}, 10^{log_hi:.3f}]", flush=True)

        step_cfg = dict(config)
        step_cfg["trans_hypparams"] = dict(step_cfg["trans_hypparams"])
        step_cfg["trans_hypparams"]["kappa"] = kappa_val

        rb.KPMSStateCache.clear()
        rb.precompute_optimal_batch_config(sub_data, verbose=False)

        model = batched_init_model_by_video(sub_data, pca, step_cfg, seed=404 + step)
        model = kpms.update_hypparams(model, kappa=float(kappa_val))

        apply_kpms_checkpoint_patch()
        rb.apply_optimizations()

        t_step_start = time.time()
        model_name = f"probe_{arm}_ext_step{step}"

        # 5 AR iters
        model, _ = kpms.fit_model(
            model, sub_data, sub_meta,
            project_dir=str(probe_base_dir),
            model_name=model_name,
            ar_only=True,
            start_iter=0,
            num_iters=5,
            parallel_message_passing=False,
            save_every_n_iters=None,
            generate_progress_plots=False,
            verbose=False,
        )

        # 10 SLDS iters
        model, _ = kpms.fit_model(
            model, sub_data, sub_meta,
            project_dir=str(probe_base_dir),
            model_name=model_name,
            ar_only=False,
            start_iter=5,
            num_iters=15,
            parallel_message_passing=False,
            save_every_n_iters=None,
            generate_progress_plots=False,
            verbose=False,
        )

        step_elapsed = time.time() - t_step_start

        z = np.array(model["states"]["z"])
        median_frames = compute_median_duration(z)
        median_ms = (median_frames / 60.0) * 1000.0 if np.isfinite(median_frames) else float("nan")
        diff = abs(median_frames - TARGET_FRAMES) if np.isfinite(median_frames) else float("inf")

        print(f"  Result: median dwell = {median_frames:.2f} frames ({median_ms:.1f} ms) | Target diff: {diff:.2f} frames ({step_elapsed:.1f}s)", flush=True)

        step_info = {
            "step": step,
            "kappa": kappa_val,
            "log10_kappa": log_curr,
            "median_frames": median_frames,
            "median_ms": median_ms,
            "diff_from_target": diff,
            "elapsed_sec": step_elapsed
        }
        history.append(step_info)

        if diff < best_diff:
            best_diff = diff
            best_step = step_info

        # Check convergence tolerance
        if np.isfinite(median_frames) and diff <= TOL_FRAMES:
            print(f"\n🎯 [ARM: {arm.upper()}] CONVERGED WITHIN TOLERANCE ({diff:.2f} <= {TOL_FRAMES}) at kappa = {kappa_val:.3e}!")
            print(f"  Median duration: {median_frames:.2f} frames ({median_ms:.1f} ms)")
            break

        # Bisection update with 42% step fraction
        if median_frames < TARGET_FRAMES:
            log_lo = max(log_lo, log_curr)
            log_curr = log_lo + STEP_FRACTION * (log_hi - log_lo)
            print(f"  Dwell too short ({median_frames:.2f} < {TARGET_FRAMES}) -> increasing kappa towards log_hi={log_hi:.2f}")
        else:
            log_hi = min(log_hi, log_curr)
            log_curr = log_hi - STEP_FRACTION * (log_hi - log_lo)
            print(f"  Dwell too long ({median_frames:.2f} > {TARGET_FRAMES}) -> decreasing kappa towards log_lo={log_lo:.2f}")

        log_curr = float(np.clip(log_curr, log_lo, log_hi))

        # Cleanup probe checkpoints
        p_dir = probe_base_dir / model_name
        if p_dir.exists():
            shutil.rmtree(p_dir, ignore_errors=True)

    if probe_base_dir.exists():
        shutil.rmtree(probe_base_dir, ignore_errors=True)

    # Free memory
    del full_data, full_meta, sub_data, sub_meta, model
    return best_step["kappa"], best_step, history


def main():
    print("=" * 80)
    print("🚀 AUTOMATED PIPELINE: EXTENDED 21KP_STATIC SCAN + PRODUCTION TRAINING")
    print("=" * 80, flush=True)

    # 1. Run the extended scan for 21kp_static
    best_kappa, best_step, history = run_extended_scan_21kp_static()

    # Load existing scan JSON and update 21kp_static
    if SCAN_JSON_PATH.exists():
        with open(SCAN_JSON_PATH, "r") as f:
            scan_data = json.load(f)
    else:
        scan_data = {}

    scan_data["21kp_static"] = {
        "optimal_scout_kappa": best_kappa,
        "optimal_scout_median_frames": best_step["median_frames"],
        "optimal_scout_median_ms": best_step["median_ms"],
        "production_kappa_3x": best_kappa * 3.0,
        "history": history
    }

    with open(SCAN_JSON_PATH, "w") as f:
        json.dump(scan_data, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ UPDATED KAPPA SCAN SUMMARY ACROSS ALL ARMS:")
    for arm, res in scan_data.items():
        print(f"  {arm:14s}: Scout Kappa* = {res['optimal_scout_kappa']:.3e} ({res['optimal_scout_median_frames']:.1f} fr, {res['optimal_scout_median_ms']:.1f} ms) -> Production (3x) = {res['production_kappa_3x']:.3e}")
    print("=" * 80, flush=True)

    # 2. Production Training: Run each arm in an isolated subprocess
    print("\n" + "#" * 80)
    print("  PHASE 2: FULL 10-SESSION PRODUCTION TRAINING (3x KAPPA)")
    print("#" * 80, flush=True)

    arms_to_run = ["17kp", "21kp_static", "21kp_dynamic"]
    for arm in arms_to_run:
        prod_k = scan_data[arm]["production_kappa_3x"]
        print(f"\n>>> LAUNCHING SUBPROCESS PRODUCTION FOR {arm.upper()} (SLDS Kappa = {prod_k:.3e}) <<<", flush=True)
        cmd = [
            PYTHON_BIN, "-u", str(RUNNER_SCRIPT),
            "--arms", arm,
            "--num-ar-iters", "50",
            "--num-slds-iters", "150",
            "--init-kappa", "1.0e6",
            "--slds-kappa", str(prod_k),
            "--save-every-n", "25",
            "--model-suffix", "calibrated400",
            "--seed", "404"
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"❌ Error: Production run for {arm} failed with return code {ret.returncode}")
            sys.exit(ret.returncode)
        print(f"✅ Production run for {arm} completed successfully!")

    # 3. Head-to-Head Comparative Evaluation
    print("\n" + "#" * 80)
    print("  PHASE 3: HEAD-TO-HEAD DURATION-MATCHED EVALUATION")
    print("#" * 80, flush=True)

    eval_csv = EXP_DIR / "04_reports_and_figures/pilot_17kp_vs_21kp_calibrated400_scorecard.csv"
    eval_json = EXP_DIR / "04_reports_and_figures/pilot_17kp_vs_21kp_calibrated400_metrics.json"

    eval_cmd = [
        PYTHON_BIN, str(EVAL_SCRIPT),
        "--suffix", "calibrated400",
        "--output-csv", str(eval_csv),
        "--output-json", str(eval_json)
    ]
    subprocess.run(eval_cmd)
    print("\n🎉 ALL TASKS COMPLETED!")


if __name__ == "__main__":
    main()
