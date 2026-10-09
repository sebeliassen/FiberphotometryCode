#!/usr/bin/env python3
"""
run_kappa_scan_and_production_pipeline.py
=========================================
End-to-end automated pipeline:
  1. Phase 1: Kappa Scan across arms (17kp, 21kp_static, 21kp_dynamic)
     - Data: 2 representative pilot sessions
     - Regimen: 5 AR + 10 SLDS = 15 iterations per probe
     - Target duration: 24.0 frames (400ms @ 60 fps) +/- 2.0 frames
     - Bounds: 10^2 to 10^16 (log10 in [2.0, 16.0])
     - Bisection search with conservative 42% step fraction
  2. Phase 2: Production Training (All 10 Sessions)
     - Applies 3x optimal scout kappa (kappa_prod = 3.0 * kappa_scout)
     - Regimen: 50 AR + 150 SLDS = 200 iterations per arm
     - Saves checkpoints every 25 iterations, extracts results.h5
     - Evaluates consecutive-iteration ARI convergence
  3. Phase 3: Head-to-Head Comparative Evaluation
     - Evaluates duration-matched models and generates master scorecard
"""

import os
import sys
import time
import json
import shutil
import pickle
from pathlib import Path
from itertools import groupby
import numpy as np
import pandas as pd
import h5py

# Unbuffered live stdout/stderr
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Headless & hardware flags
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

import jax
import keypoint_moseq as kpms
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

KPMS_UTILS_DIR = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll"
if str(KPMS_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(KPMS_UTILS_DIR))

import utils.resample_batching as rb
from utils.batch_init import batched_init_model_by_video
from syllable_generation.syllabizers.keypoint_moseq_syll.utils.patch_checkpoint import apply_kpms_checkpoint_patch

TARGET_FPS = 60
SCOUT_SIDS = [
    "2026-03-05T11_01_01_bottom_left",
    "2026-03-05T12_46_18_top_left"
]
ARMS = ["17kp", "21kp_static", "21kp_dynamic"]

TARGET_FRAMES = 24.0  # 400ms at 60 fps
TOL_FRAMES = 2.0     # 22.0 to 26.0 frames
STEP_FRACTION = 0.42 # conservative bisection step
MAX_SCAN_STEPS = 8

LOG_LO = 2.0   # 1e2
LOG_HI = 16.0  # 1e16
LOG_START = 9.0 # 1e9 midpoint


def compute_median_duration(z):
    """Computes median bout length in frames from syllable states."""
    durations = []
    if z.ndim == 1:
        z = z[None, :]
    for row in z:
        for val, grp in groupby(row):
            if val >= 0:
                durations.append(len(list(grp)))
    return float(np.median(durations)) if len(durations) > 0 else float("nan")


def slice_scout_data(full_data, full_meta, scout_sids):
    """Extracts scout sessions directly from preprocessed batch arrays."""
    keys = full_meta[0]
    scout_mask = np.array([any(sid in str(k) for sid in scout_sids) for k in keys])
    scout_idx = np.where(scout_mask)[0]
    
    sub_data = {k: v[scout_idx] for k, v in full_data.items()}
    sub_meta = (keys[scout_idx], full_meta[1][scout_idx])
    return sub_data, sub_meta


def compute_consecutive_ari_convergence(checkpoint_path):
    """Computes pairwise ARI across all saved checkpoint snapshots."""
    try:
        with h5py.File(checkpoint_path, "r") as f:
            if "model_snapshots" not in f:
                return {}
            snapshots = np.sort([int(i) for i in f["model_snapshots"]])
            if len(snapshots) < 2:
                return {}

            z_seqs = {}
            for s in snapshots:
                grp_path = f"model_snapshots/{s}/states/z"
                if grp_path in f:
                    z_seqs[s] = f[grp_path][:].ravel()

        avail_snaps = sorted(z_seqs.keys())
        if len(avail_snaps) < 2:
            return {}

        ari_results = {}
        for i in range(len(avail_snaps) - 1):
            s_prev = avail_snaps[i]
            s_curr = avail_snaps[i + 1]
            z_prev = z_seqs[s_prev]
            z_curr = z_seqs[s_curr]
            valid_mask = (z_prev >= 0) & (z_curr >= 0)
            ari = float(adjusted_rand_score(z_prev[valid_mask], z_curr[valid_mask]))
            flip_rate = float(np.mean(z_prev[valid_mask] != z_curr[valid_mask])) * 100.0
            ari_results[f"iter_{s_prev}_to_{s_curr}"] = {
                "from_iter": int(s_prev),
                "to_iter": int(s_curr),
                "ari": ari,
                "label_change_pct": flip_rate
            }
        return ari_results
    except Exception as e:
        print(f"  ⚠️ Warning: ARI convergence evaluation encountered: {e}")
        return {}


def run_kappa_scan(arm, sub_data, sub_meta, config, pca, arm_dir):
    """
    Executes the 42% step bisection scan over [1e2, 1e16] for a single arm.
    """
    print("\n" + "=" * 80)
    print(f"🔍 STARTING KAPPA SCAN FOR ARM: {arm.upper()}")
    print(f"  Target Duration: {TARGET_FRAMES} frames (400ms @ 60fps) +/- {TOL_FRAMES} frames")
    print(f"  Bounds: [10^{LOG_LO:.1f}, 10^{LOG_HI:.1f}] | Start: 10^{LOG_START:.1f}")
    print(f"  Step Fraction: {STEP_FRACTION:.2f} (Conservative Bisection)")
    print(f"  Scout Batches: {len(sub_data['Y'])} batches across {len(SCOUT_SIDS)} sessions")
    print("=" * 80, flush=True)

    log_lo = LOG_LO
    log_hi = LOG_HI
    log_curr = LOG_START

    history = []
    best_step = None
    best_diff = float("inf")

    probe_base_dir = arm_dir / "kappa_probes"
    probe_base_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, MAX_SCAN_STEPS + 1):
        kappa_val = float(10**log_curr)
        print(f"\n--- [ARM: {arm.upper()}] SCAN STEP {step}/{MAX_SCAN_STEPS} ---")
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
        model_name = f"probe_{arm}_step{step}"

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

        # Measure duration
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
            print(f"\n🎯 [ARM: {arm.upper()}] CONVERGED within tolerance ({diff:.2f} <= {TOL_FRAMES}) at kappa = {kappa_val:.3e}!")
            break

        # Bisection update with 42% step fraction
        if median_frames < TARGET_FRAMES:
            # Dwell too short -> increase kappa
            log_lo = max(log_lo, log_curr)
            log_curr = log_lo + STEP_FRACTION * (log_hi - log_lo)
            print(f"  Dwell too short ({median_frames:.2f} < {TARGET_FRAMES}) -> increasing kappa towards log_hi={log_hi:.2f}")
        else:
            # Dwell too long -> decrease kappa
            log_hi = min(log_hi, log_curr)
            log_curr = log_hi - STEP_FRACTION * (log_hi - log_lo)
            print(f"  Dwell too long ({median_frames:.2f} > {TARGET_FRAMES}) -> decreasing kappa towards log_lo={log_lo:.2f}")

        log_curr = float(np.clip(log_curr, log_lo, log_hi))

        # Cleanup probe checkpoints
        p_dir = probe_base_dir / model_name
        if p_dir.exists():
            shutil.rmtree(p_dir, ignore_errors=True)

    # Clean probe base dir
    if probe_base_dir.exists():
        shutil.rmtree(probe_base_dir, ignore_errors=True)

    print(f"\n🏆 Best scout kappa for {arm.upper()}: {best_step['kappa']:.3e} -> {best_step['median_frames']:.2f} frames ({best_step['median_ms']:.1f} ms)")
    return best_step["kappa"], best_step, history


def run_production_training(arm, full_data, full_meta, config, pca, arm_dir, slds_kappa, seed=404):
    """
    Trains the full 10-session dataset for 200 iterations using the 3x calibrated kappa.
    """
    model_suffix = "calibrated400"
    model_name = f"model_{arm}_{model_suffix}_seed{seed}"
    project_dir = arm_dir / model_name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous run artifacts
    for old_f in [project_dir / "results.h5", project_dir / "checkpoint.h5", project_dir / "ari_convergence.json"]:
        if old_f.exists():
            old_f.unlink()

    print("\n" + "=" * 80)
    print(f"🚀 LAUNCHING FULL PRODUCTION TRAINING: {arm.upper()}")
    print(f"  Model Name:          {model_name}")
    print(f"  SLDS Kappa (3x):     {slds_kappa:.3e}")
    print(f"  Total Batches:       {len(full_data['Y'])}")
    print(f"  Regimen:             50 AR + 150 SLDS = 200 Gibbs Iterations")
    print("=" * 80, flush=True)

    rb.KPMSStateCache.clear()
    rb.precompute_optimal_batch_config(full_data, verbose=False)

    model = batched_init_model_by_video(full_data, pca, config, seed=seed)
    model = kpms.update_hypparams(model, kappa=1.0e6)

    apply_kpms_checkpoint_patch()
    rb.apply_optimizations()

    t_train_start = time.time()

    # 50 AR iters
    print(f"\n>>> Running AR Phase (iters 0 -> 50)...", flush=True)
    model, _ = kpms.fit_model(
        model, full_data, full_meta,
        project_dir=str(arm_dir),
        model_name=model_name,
        ar_only=True,
        start_iter=0,
        num_iters=50,
        parallel_message_passing=False,
        save_every_n_iters=25,
        generate_progress_plots=False,
        verbose=True,
    )

    # Transition to calibrated SLDS kappa
    print(f"\n>>> Transitioning to SLDS Phase (calibrated kappa = {slds_kappa:.3e})...", flush=True)
    model = kpms.update_hypparams(model, kappa=float(slds_kappa))
    rb.KPMSStateCache.clear()
    rb.precompute_optimal_batch_config(full_data, model=model, verbose=False)

    # 150 SLDS iters
    print(f"\n>>> Running SLDS Phase (iters 50 -> 200)...", flush=True)
    model, _ = kpms.fit_model(
        model, full_data, full_meta,
        project_dir=str(arm_dir),
        model_name=model_name,
        ar_only=False,
        start_iter=50,
        num_iters=200,
        parallel_message_passing=False,
        save_every_n_iters=25,
        generate_progress_plots=False,
        verbose=True,
    )

    train_time_min = (time.time() - t_train_start) / 60.0
    print(f"\n[TRAINING FINISHED] Total time: {train_time_min:.2f} minutes.", flush=True)

    # Extract results
    print("  Extracting results...", flush=True)
    res_path = project_dir / "results.h5"
    if res_path.exists():
        res_path.unlink()
    kpms.extract_results(model, full_meta, str(arm_dir), model_name, save_results=True)
    print(f"  Saved results to {res_path}")

    # Evaluate Convergence ARI
    chk_path = project_dir / "checkpoint.h5"
    if chk_path.exists():
        print("\n  Evaluating consecutive-iteration ARI convergence...", flush=True)
        ari_convergence = compute_consecutive_ari_convergence(chk_path)
        for step, m in ari_convergence.items():
            print(f"    {step}: ARI = {m['ari']:.4f} | Label Flip = {m['label_change_pct']:.1f}%")
        conv_file = project_dir / "ari_convergence.json"
        conv_file.write_text(json.dumps(ari_convergence, indent=2))
        print(f"  Saved convergence report to {conv_file}")

    return str(res_path), train_time_min


def main():
    print("=" * 80)
    print("⚡ AUTOMATED MULTI-ARM KAPPA SCAN & PRODUCTION BENCHMARK")
    print(f"Arms:               {ARMS}")
    print(f"Scout Sessions:     {SCOUT_SIDS}")
    print(f"Target Duration:    {TARGET_FRAMES} frames (400ms @ 60fps) +/- {TOL_FRAMES} frames")
    print(f"Search Bounds:      [10^{LOG_LO:.1f}, 10^{LOG_HI:.1f}]")
    print(f"Step Fraction:      {STEP_FRACTION:.2f} (Conservative Bisection)")
    print(f"Production Scaling: 3.0x Scout Kappa")
    print("=" * 80, flush=True)

    pipeline_start = time.time()
    scan_summaries = {}
    prod_results = {}

    # -------------------------------------------------------------------------
    # PHASE 1: KAPPA SCAN ACROSS ALL ARMS
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("  PHASE 1: RUNNING KAPPA SCANS (2 SESSIONS PER ARM)")
    print("#" * 80, flush=True)

    arm_data = {}

    for arm in ARMS:
        arm_dir = EXP_DIR / f"pilot_comparison_{arm}"
        with open(arm_dir / "preprocessed_data.pkl", "rb") as f:
            full_data, full_meta = pickle.load(f)
        with open(arm_dir / "pca.p", "rb") as f:
            pca = pickle.load(f)
        config = kpms.load_config(str(arm_dir))

        sub_data, sub_meta = slice_scout_data(full_data, full_meta, SCOUT_SIDS)
        arm_data[arm] = (full_data, full_meta, config, pca, arm_dir)

        best_kappa, best_step, history = run_kappa_scan(
            arm=arm,
            sub_data=sub_data,
            sub_meta=sub_meta,
            config=config,
            pca=pca,
            arm_dir=arm_dir
        )

        scan_summaries[arm] = {
            "optimal_scout_kappa": best_kappa,
            "optimal_scout_median_frames": best_step["median_frames"],
            "optimal_scout_median_ms": best_step["median_ms"],
            "production_kappa_3x": best_kappa * 3.0,
            "history": history
        }

    # Save scan results to JSON
    scan_json_path = EXP_DIR / "04_reports_and_figures/kappa_scan_results.json"
    scan_json_path.parent.mkdir(parents=True, exist_ok=True)
    scan_json_path.write_text(json.dumps(scan_summaries, indent=2))
    print(f"\n✅ All kappa scans completed! Saved scan results to {scan_json_path}")

    print("\n--- Summary of Calibrated Kappas ---")
    for arm, res in scan_summaries.items():
        print(f"  {arm:14s}: Scout Kappa* = {res['optimal_scout_kappa']:.3e} ({res['optimal_scout_median_frames']:.1f} fr, {res['optimal_scout_median_ms']:.1f} ms) -> Production Kappa (3x) = {res['production_kappa_3x']:.3e}")

    # -------------------------------------------------------------------------
    # PHASE 2: FULL PRODUCTION RETRAINING (ALL 10 SESSIONS @ 3x KAPPA)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("  PHASE 2: FULL 10-SESSION PRODUCTION TRAINING (3x KAPPA)")
    print("#" * 80, flush=True)

    for arm in ARMS:
        full_data, full_meta, config, pca, arm_dir = arm_data[arm]
        prod_kappa = scan_summaries[arm]["production_kappa_3x"]

        res_path, run_min = run_production_training(
            arm=arm,
            full_data=full_data,
            full_meta=full_meta,
            config=config,
            pca=pca,
            arm_dir=arm_dir,
            slds_kappa=prod_kappa,
            seed=404
        )
        prod_results[arm] = {
            "results_path": res_path,
            "runtime_min": run_min,
            "prod_kappa": prod_kappa
        }

    # -------------------------------------------------------------------------
    # PHASE 3: COMPARATIVE EVALUATION & MASTER SCORECARD
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("  PHASE 3: HEAD-TO-HEAD DURATION-MATCHED EVALUATION")
    print("#" * 80, flush=True)

    eval_script = EXP_DIR / "03_evaluation_and_eml/evaluate_17kp_vs_21kp_comparison.py"
    eval_csv = EXP_DIR / "04_reports_and_figures/pilot_17kp_vs_21kp_calibrated400_scorecard.csv"
    eval_json = EXP_DIR / "04_reports_and_figures/pilot_17kp_vs_21kp_calibrated400_metrics.json"

    cmd = (
        f"{sys.executable} {eval_script} "
        f"--suffix calibrated400 "
        f"--output-csv {eval_csv} "
        f"--output-json {eval_json}"
    )
    print(f"Running evaluation: {cmd}\n")
    os.system(cmd)

    total_pipeline_time = (time.time() - pipeline_start) / 60.0
    print("\n" + "=" * 80)
    print(f"🎉 PIPELINE FULLY COMPLETED IN {total_pipeline_time:.2f} MINUTES!")
    print(f"  Scorecard CSV: {eval_csv}")
    print(f"  Metrics JSON:   {eval_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
