#!/usr/bin/env python3
"""
run_10session_static_calibration.py
===================================
Orchestrates targeted 10-session kappa calibration for `21kp_static` and creates
dorsal syllable instance grid movies:

1. Kappa Calibration (10 Sessions, 20 AR + 80 SLDS = 100 iters):
   - Target dwell: 24.0 frames (400ms @ 60fps) +/- 2.0 frames (22.0 to 26.0 frames).
   - Starts at log10(kappa) = 15.0 (1.0e15), interpolated from 1.5e12 (16 fr) and 9.27e20 (41 fr).
   - Uses conservative bisection (step fraction 0.42) until within tolerance.
   - Each probe is run in an isolated Python subprocess for 100% clean JAX GPU VRAM.

2. Full-Cohort Kappa Scaling:
   - Records optimal 10-session scout kappa (kappa*).
   - Computes recommended 118-session cohort kappa (1.5x to 2.0x kappa*).

3. Dorsal Syllable Instances:
   - Invokes CPU-isolated grid movie generator using kpms.generate_grid_movies
     with verified 1:1 video frame mappings to karen_project/video_files/*.avi.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from itertools import groupby
from pathlib import Path

import h5py
import numpy as np

# Live unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
RUNNER_SCRIPT = EXP_DIR / "02_training_runners/run_pilot_17kp_vs_21kp_comparison.py"
ARM_DIR = EXP_DIR / "pilot_comparison_21kp_static"
OUT_JSON = EXP_DIR / "04_reports_and_figures/static_21kp_10sess_calibration_results.json"
GPU_PYTHON = "/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
CPU_PYTHON = "/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
GRID_MOVIE_SCRIPT = EXP_DIR / "02_training_runners/generate_dorsal_grid_movies.py"

TARGET_FRAMES = 24.0  # 400 ms @ 60 fps
TOL_FRAMES = 2.0     # 22.0 to 26.0 frames
STEP_FRACTION = 0.42 # conservative bisection step fraction
MAX_STEPS = 8


def compute_median_duration(results_h5_path):
    """Computes median duration in frames across all syllables >= 0."""
    durations = []
    with h5py.File(results_h5_path, "r") as f:
        for k in f.keys():
            if "syllable" in f[k]:
                z = f[k]["syllable"][:]
                for val, grp in groupby(z):
                    if val >= 0:
                        durations.append(len(list(grp)))
    return float(np.median(durations)) if len(durations) > 0 else float("nan")


def run_probe_subprocess(kappa_val, step_idx, seed=404):
    """Runs a single 100-iter (20 AR + 80 SLDS) probe in a fresh Python process."""
    model_suffix = f"_calib100_step{step_idx}"
    model_dir_name = f"model_21kp_static_{model_suffix}_seed{seed}"
    model_dir = ARM_DIR / model_dir_name
    results_h5 = model_dir / "results.h5"

    if results_h5.exists():
        print(f"\n[CACHE HIT] Found completed probe results at {results_h5}!", flush=True)
        elapsed = 0.0
    else:
        cmd = [
            GPU_PYTHON,
            str(RUNNER_SCRIPT),
            "--arms", "21kp_static",
            "--num-ar-iters", "20",
            "--num-slds-iters", "80",
            "--slds-kappa", str(kappa_val),
            "--save-every-n", "25",
            "--seed", str(seed),
            "--model-suffix", model_suffix
        ]
        print(f"\n[ORCHESTRATOR] Launching subprocess probe step {step_idx}:")
        print(f"  Command: {' '.join(cmd)}", flush=True)

        t0 = time.time()
        env = os.environ.copy()
        env["BROWSER"] = "/bin/true"
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
        env["CUDA_VISIBLE_DEVICES"] = "0"

        proc = subprocess.run(cmd, env=env, text=True, capture_output=False)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            raise RuntimeError(f"Subprocess probe failed with exit code {proc.returncode}")

        if not results_h5.exists():
            raise FileNotFoundError(f"Expected results.h5 not found at {results_h5}")

    med_frames = compute_median_duration(results_h5)
    med_ms = (med_frames / 60.0) * 1000.0 if np.isfinite(med_frames) else float("nan")
    diff = abs(med_frames - TARGET_FRAMES) if np.isfinite(med_frames) else float("inf")

    return {
        "step": step_idx,
        "kappa": kappa_val,
        "log10_kappa": float(np.log10(kappa_val)),
        "median_frames": med_frames,
        "median_ms": med_ms,
        "diff_from_target": diff,
        "elapsed_min": elapsed / 60.0,
        "model_dir": str(model_dir),
        "results_h5": str(results_h5),
    }


def main():
    print("=" * 80)
    print("🎯 STARTING 10-SESSION KAPPA CALIBRATION FOR 21KP_STATIC")
    print(f"  Target Dwell:     {TARGET_FRAMES:.1f} frames (400 ms @ 60fps) +/- {TOL_FRAMES:.1f} frames")
    print(f"  Acceptance Range: [{TARGET_FRAMES - TOL_FRAMES:.1f}, {TARGET_FRAMES + TOL_FRAMES:.1f}] frames ({ (TARGET_FRAMES - TOL_FRAMES)/60*1000:.1f} to { (TARGET_FRAMES + TOL_FRAMES)/60*1000:.1f} ms)")
    print("  Regimen:          20 AR + 80 SLDS = 100 iterations on all 10 sessions (~1.74M frames)")
    print("  Initial Kappa:    1.000e+15 (log10 = 15.000)")
    print(f"  Step Fraction:    {STEP_FRACTION:.2f}")
    print("=" * 80, flush=True)

    log_lo = 12.0
    log_hi = 21.0
    log_curr = 15.0  # Interpolated prior estimate

    history = []
    winning_probe = None

    for step in range(1, MAX_STEPS + 1):
        kappa_val = float(10**log_curr)
        print("\n================================================================================")
        print(f"📍 SCAN STEP {step}/{MAX_STEPS}: Testing kappa = {kappa_val:.3e} (log10 = {log_curr:.3f})")
        print(f"   Active bounds: [10^{log_lo:.3f}, 10^{log_hi:.3f}]")
        print("================================================================================", flush=True)

        res = run_probe_subprocess(kappa_val, step, seed=404)
        history.append(res)

        print(f"\n📊 STEP {step} RESULT:")
        print(f"   Median Duration: {res['median_frames']:.2f} frames ({res['median_ms']:.1f} ms)")
        print(f"   Diff from 24fr:  {res['diff_from_target']:.2f} frames")
        print(f"   Runtime:         {res['elapsed_min']:.2f} minutes", flush=True)

        if res["diff_from_target"] <= TOL_FRAMES:
            print(f"\n🎯 CONVERGENCE ACHIEVED ON STEP {step}!")
            print(f"   Median duration {res['median_frames']:.2f} frames is within tolerance [{TARGET_FRAMES - TOL_FRAMES:.1f}, {TARGET_FRAMES + TOL_FRAMES:.1f}]!")
            winning_probe = res
            break

        # Bisect with 0.42 step fraction
        if res["median_frames"] < TARGET_FRAMES:
            print(f"   Dwell time too fast ({res['median_frames']:.2f} < {TARGET_FRAMES:.1f}). Stepping kappa UP.")
            log_lo = log_curr
            log_curr = log_curr + STEP_FRACTION * (log_hi - log_curr)
        else:
            print(f"   Dwell time too slow ({res['median_frames']:.2f} > {TARGET_FRAMES:.1f}). Stepping kappa DOWN.")
            log_hi = log_curr
            log_curr = log_curr - STEP_FRACTION * (log_curr - log_lo)

    if winning_probe is None:
        # Pick probe closest to target
        print("\n⚠️ Max scan steps reached without strict convergence; selecting best probe.")
        winning_probe = min(history, key=lambda x: x["diff_from_target"])

    opt_kappa = winning_probe["kappa"]
    cohort_kappa_1p5x = opt_kappa * 1.5
    cohort_kappa_2p0x = opt_kappa * 2.0

    print("\n" + "=" * 80)
    print("🏆 FINAL CALIBRATION SUMMARY:")
    print(f"  Winning Probe:     Step {winning_probe['step']}")
    print(f"  Calibrated Kappa:  {opt_kappa:.4e} (log10 = {winning_probe['log10_kappa']:.3f})")
    print(f"  Median Duration:   {winning_probe['median_frames']:.2f} frames ({winning_probe['median_ms']:.1f} ms)")
    print(f"  Model Directory:   {winning_probe['model_dir']}")
    print(f"  Recommended Cohort Kappa (1.5x - 2.0x): {cohort_kappa_1p5x:.4e} to {cohort_kappa_2p0x:.4e}")
    print("=" * 80, flush=True)

    # Designate final model directory
    final_model_dir = ARM_DIR / "model_21kp_static_calibrated100_final"
    if final_model_dir.exists():
        if final_model_dir.is_symlink():
            final_model_dir.unlink()
        else:
            shutil.rmtree(final_model_dir)
    os.symlink(winning_probe["model_dir"], final_model_dir)
    print(f"  Linked {final_model_dir} -> {winning_probe['model_dir']}")

    # Save summary json
    summary_data = {
        "arm": "21kp_static",
        "regimen": "20 AR + 80 SLDS (100 iters)",
        "num_sessions": 10,
        "target_frames": TARGET_FRAMES,
        "target_ms": 400.0,
        "tolerance_frames": TOL_FRAMES,
        "optimal_10session_kappa": opt_kappa,
        "optimal_median_frames": winning_probe["median_frames"],
        "optimal_median_ms": winning_probe["median_ms"],
        "recommended_cohort_kappa_1.5x": cohort_kappa_1p5x,
        "recommended_cohort_kappa_2.0x": cohort_kappa_2p0x,
        "winning_model_dir": winning_probe["model_dir"],
        "final_symlink_dir": str(final_model_dir),
        "history": history,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Saved calibration results to {OUT_JSON}")

    # Launch Dorsal Syllable Grid Movies Generation
    print("\n" + "=" * 80)
    print("🎬 LAUNCHING DORSAL SYLLABLE INSTANCES (GRID MOVIES) GENERATION")
    print("=" * 80, flush=True)
    grid_cmd = [
        CPU_PYTHON,
        str(GRID_MOVIE_SCRIPT),
        "--model-dir", str(final_model_dir),
        "--top-n-syllables", "15"
    ]
    print(f"  Command: {' '.join(grid_cmd)}", flush=True)
    grid_proc = subprocess.run(grid_cmd, capture_output=False)
    if grid_proc.returncode != 0:
        print(f"⚠️ Warning: Grid movie generator exited with code {grid_proc.returncode}")
    else:
        print("✅ Dorsal syllable grid movies generated successfully!")


if __name__ == "__main__":
    main()
