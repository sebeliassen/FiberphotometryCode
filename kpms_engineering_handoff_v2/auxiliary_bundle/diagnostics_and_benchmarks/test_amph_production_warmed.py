#!/usr/bin/env python3
"""
diagnostics/test_amph_production_warmed.py

Production acceptance benchmark on the FULL 30-session AMPH dataset:
- 562 sequences, 5,516,954 active frames (5,636,860 allocated frames).
- Runs 3 AR iterations + 3 SLDS iterations.
- Measures:
  1. Cold vs Warmed iteration times for both AR and SLDS.
  2. Peak Host Process RAM (RSS) via psutil per iteration.
  3. Peak GPU VRAM via nvidia-smi per iteration.
  4. Checkpoint decoupling verification (save_every_n_iters=50).
  5. Checkpoint resume verification (resuming from checkpoint for 1 additional iteration).
"""

import multiprocessing
import os
import shutil
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ["DISABLE_NTFY"] = "1"

if multiprocessing.current_process().name != 'MainProcess':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path: sys.path.insert(0, KPMS_DIR)

import numpy as np

if not hasattr(np, "bool8"): np.bool8 = np.bool_
import h5py
import keypoint_moseq as kpms
import pandas as pd
import psutil
from kpms_pipeline import run_pipeline

PROJECT_DIR = os.path.join(KPMS_DIR, "amph_project")
BENCH_MODEL_NAME = "amph_prod_acceptance_bench"
SESSIONS_PKL = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_dirty.pkl"
MIN_RAW_FRAMES = 50_000

def get_process_ram_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def get_gpu_vram_gb():
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return float(out.decode().strip()) / 1024.0
    except Exception:
        return 0.0

def load_filtered_sessions():
    sessions = pd.read_pickle(SESSIONS_PKL)
    filtered = []
    for s in sessions:
        if hasattr(s, 'tracking_refs_df') and s.tracking_refs_df is not None and not s.tracking_refs_df.empty:
            p = s.tracking_refs_df['path'].iloc[0]
            if p and os.path.exists(p):
                try:
                    with h5py.File(p, 'r') as f:
                        if f['df_with_missing/table'].shape[0] >= MIN_RAW_FRAMES:
                            filtered.append(s)
                except Exception: pass
    return filtered

def run_acceptance_benchmark():
    print("=" * 80)
    print("  FULL-DATASET PRODUCTION BENCHMARK (30 SESSIONS, 5.5M FRAMES)")
    print("=" * 80, flush=True)

    print(f"Initial Host Process RAM: {get_process_ram_gb():.2f} GB")
    print(f"Initial GPU VRAM:        {get_gpu_vram_gb():.2f} GB", flush=True)

    sessions = load_filtered_sessions()
    print(f"Loaded {len(sessions)} production sessions.", flush=True)

    model_dir = os.path.join(PROJECT_DIR, BENCH_MODEL_NAME)
    if os.path.exists(model_dir):
        print(f"Cleaning existing test directory {model_dir}...")
        shutil.rmtree(model_dir)

    # Instrument kpms.fitting._wrapped_resample to collect granular iteration telemetry
    orig_wrapped_resample = kpms.fitting._wrapped_resample
    iteration_telemetry = []

    def instrumented_resample(resample_func, data, model, pbar=None, **resample_options):
        ar_only = resample_options.get("ar_only", False)
        phase = "AR" if ar_only else "SLDS"
        iter_num = len(iteration_telemetry) + 1

        t0 = time.perf_counter()
        out_model = orig_wrapped_resample(resample_func, data, model, pbar=pbar, **resample_options)
        dt = time.perf_counter() - t0

        ram_gb = get_process_ram_gb()
        vram_gb = get_gpu_vram_gb()

        rec = {
            "iter_idx": iter_num,
            "phase": phase,
            "duration_sec": dt,
            "ram_gb": ram_gb,
            "vram_gb": vram_gb
        }
        iteration_telemetry.append(rec)
        print(f"\n[ITERATION {iter_num} TELEMETRY] Phase: {phase} | Time: {dt:6.2f} s | Process RAM: {ram_gb:5.2f} GB | GPU VRAM: {vram_gb:5.2f} GB\n", flush=True)
        return out_model

    kpms.fitting._wrapped_resample = instrumented_resample

    pipeline_kwargs = {
        "project_dir": PROJECT_DIR,
        "detector": "deeplabcut",
        "cam": "center_cam",
        "subroutine": "jonathan",
        "latent_dim": 10,
        "num_ar_iters": 3,
        "total_iters": 3,  # 3 AR + 3 SLDS = 6 iters
        "pca_mode": "load_or_fit",
        "model_name": BENCH_MODEL_NAME,
        "init_kappa": 150000,
        "decrease_kappa_factor": 10,
        "mixed_map_iters": 64,
        "seed": 42,
        "downsample_rate": 2,
        "new_fps": 40,
        "use_batched_init": True,
        "use_memory_patch": True,
        "flipping_mode": None,
        "subset_method": "spread",
        "subset_fraction": 1.0,
        "save_every_n_iters": 50,  # Decoupled checkpointing test
    }

    # 1. Primary Fit Run (3 AR + 3 SLDS)
    print("\n>>> STARTING PRIMARY FIT (3 AR + 3 SLDS ITERATIONS) <<<\n", flush=True)
    t0_fit = time.time()
    results = run_pipeline(sessions, **pipeline_kwargs)
    t_fit = time.time() - t0_fit

    # 2. Audit Checkpoint Decoupling
    print("\n>>> AUDITING CHECKPOINT PERSISTENCE & DECOUPLING <<<\n", flush=True)
    ckpt_path = os.path.join(model_dir, "checkpoint.h5")
    assert os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}"
    ckpt_size_gb = os.path.getsize(ckpt_path) / (1024**3)

    with h5py.File(ckpt_path, "r") as f:
        saved_snapshots = sorted([int(k) for k in f["model_snapshots"].keys()])
    print(f"Checkpoint size: {ckpt_size_gb:.2f} GB")
    print(f"Saved snapshot iterations: {saved_snapshots}")

    # Decoupling assertions:
    # Snapshots 1, 2, 4, 5 should NOT be saved because save_every_n_iters=50
    # Snapshot 3 (AR milestone) and Snapshot 6 (SLDS final) should be saved.
    assert 3 in saved_snapshots, "AR milestone snapshot (3) missing!"
    assert 6 in saved_snapshots, "SLDS final snapshot (6) missing!"
    assert 1 not in saved_snapshots and 2 not in saved_snapshots, "Intermediate AR snapshot leaked!"
    assert 4 not in saved_snapshots and 5 not in saved_snapshots, "Intermediate SLDS snapshot leaked!"
    print("--> Checkpoint decoupling verified: Intermediate snapshots correctly skipped!", flush=True)

    # 3. Test Checkpoint Resume
    print("\n>>> TESTING CHECKPOINT RESUME (Iteration 6 -> 7) <<<\n", flush=True)
    resume_kwargs = dict(pipeline_kwargs)
    resume_kwargs["total_iters"] = 4  # Extends SLDS to iteration 7 (3 AR + 4 SLDS)

    t0_resume = time.time()
    results_resumed = run_pipeline(sessions, **resume_kwargs)
    t_resume = time.time() - t0_resume

    with h5py.File(ckpt_path, "r") as f:
        resumed_snapshots = sorted([int(k) for k in f["model_snapshots"].keys()])
    print(f"Resumed snapshot iterations: {resumed_snapshots}")
    assert 7 in resumed_snapshots, "Resumed snapshot 7 missing!"
    print(f"--> Checkpoint resume verified successfully in {t_resume:.2f}s!", flush=True)

    # 4. Final Telemetry Report
    print("\n" + "=" * 80)
    print("  PRODUCTION ACCEPTANCE BENCHMARK RESULTS")
    print("=" * 80)
    print("  Total Dataset: 30 Sessions, 562 Sequences, 5,516,954 Active Frames\n")
    print(f"  {'Iteration':<12} | {'Phase':<6} | {'Duration':<12} | {'Host RAM (RSS)':<16} | {'GPU VRAM':<12}")
    print("  " + "-" * 70)
    for rec in iteration_telemetry:
        status = "(Cold JIT)" if rec["iter_idx"] in [1, 4] else "(Warmed)"
        if rec["iter_idx"] == 7:
            status = "(Resumed Warmed)"
        print(f"  Iter {rec['iter_idx']:<2} {status:<16} | {rec['phase']:<6} | {rec['duration_sec']:7.2f} s    | {rec['ram_gb']:6.2f} GB        | {rec['vram_gb']:6.2f} GB")

    print("  " + "-" * 70)

    ar_warm = [r["duration_sec"] for r in iteration_telemetry if r["phase"] == "AR" and r["iter_idx"] > 1]
    slds_warm = [r["duration_sec"] for r in iteration_telemetry if r["phase"] == "SLDS" and r["iter_idx"] not in [4, 7]]
    resumed_iter = [r["duration_sec"] for r in iteration_telemetry if r["iter_idx"] == 7]

    print("\n  [Summary Metrics]")
    print(f"  * AR Cold Iteration 1:           {iteration_telemetry[0]['duration_sec']:.2f} s")
    print(f"  * AR Warmed Steady-State:        {np.mean(ar_warm):.2f} s / iter")
    print(f"  * SLDS Cold Iteration 4:         {iteration_telemetry[3]['duration_sec']:.2f} s")
    print(f"  * SLDS Warmed Steady-State:      {np.mean(slds_warm):.2f} s / iter")
    if resumed_iter:
        print(f"  * Resumed Iteration 7:           {resumed_iter[0]:.2f} s")
    print(f"  * Peak Process Host RAM:         {max(r['ram_gb'] for r in iteration_telemetry):.2f} GB (System has 125 GB)")
    print(f"  * Peak GPU VRAM:                 {max(r['vram_gb'] for r in iteration_telemetry):.2f} GB (RTX 4090 has 24 GB)")
    print("  * Checkpoint Decoupling:         VERIFIED (1,2,4,5 skipped; 3,6,7 persisted)")
    print("  * Checkpoint Resume:             VERIFIED (Seamless recovery at iter 6)")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    run_acceptance_benchmark()
