#!/usr/bin/env python3
"""
run_cohort_static_21kp_v2_production.py
=======================================
Full Cohort (118 Sessions) 21kp Static Multi-View Production Runner (V2):
- Trained on epoch-corrected fused inputs: data/cohort_fused_static_21kp_v2
- Full fresh training: 50 AR + 300 SLDS = 350 total iterations (Seed 101)
- Dynamic chunk scheduling (7 x 50-iteration blocks)
- Real-time ntfy.sh notifications every 50 iterations
- Amdahl 27x Banded Cholesky JAX kernels and memory optimizations
- Cryptographic provenance manifests for ingestion and PCA
- Consecutive-iteration ARI convergence analysis on saved checkpoints
"""

import os
import sys

# Live unbuffered stdout/stderr
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Hardware environment flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["USE_MEMORY_PATCH"] = "1"

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import argparse
import gc
import json
import pickle
import shutil
import signal
import time
import urllib.request
from itertools import groupby
from pathlib import Path

import h5py
import jax
import matplotlib
import pandas as pd

matplotlib.use("Agg")

import keypoint_moseq as kpms

from fpcode.common.paths import REPO_ROOT
EXPERIMENT_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
KPMS_UTILS_DIR = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll"
if str(KPMS_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(KPMS_UTILS_DIR))

import utils.resample_batching as rb
from utils.batch_init import batched_init_model_by_video

from core.cache_manifest import write_manifest
from syllable_generation.syllabizers.keypoint_moseq_syll.utils.patch_checkpoint import apply_kpms_checkpoint_patch

# Configuration constants
DATA_SOURCE_DIR = EXPERIMENT_DIR / "data/cohort_fused_static_21kp_v2"
COHORT_DIR = EXPERIMENT_DIR / "cohort_static_21kp_v2"

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

DORSAL_BPS = [
    "nose", "left_ear_tip", "right_ear_tip", "neck", "mid_back",
    "mouse_center", "mid_backend", "mid_backend2", "mid_backend3",
    "tail_base", "left_shoulder", "left_midside", "left_hip",
    "right_shoulder", "right_midside", "right_hip", "head_midpoint"
]
PAW_BPS = ["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw"]
BODYPARTS_21KP = DORSAL_BPS + PAW_BPS

TARGET_FPS = 60
MIN_SEGMENT_FRAMES = 600  # 10.0s @ 60 FPS


from fpcode.common.notify import send_ntfy


def handle_shutdown_signal(signum, frame):
    sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"\n[shutdown] Caught signal {sig_name} (PID {os.getpid()}). Flushing and alerting...", flush=True)
    send_ntfy(
        f"KPMS V2 Cohort Run Interrupted ({sig_name})",
        f"Process caught {sig_name}. Checkpoints are intact; script can resume seamlessly.",
        priority="high",
        tags="warning,pause"
    )
    sys.exit(128 + signum)


signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)


def split_dual_view_segments(df: pd.DataFrame, ventral_conf_thresh: float = 0.4):
    """Computes continuous segments where ventral keypoints are tracked."""
    scorer = df.columns.levels[0][0]
    n_frames = len(df)
    is_valid = np.zeros(n_frames, dtype=bool)

    for bp in PAW_BPS:
        col_x = (scorer, bp, "x")
        col_p = (scorer, bp, "likelihood")
        if col_x in df.columns and col_p in df.columns:
            finite = np.isfinite(df[col_x].to_numpy())
            conf = df[col_p].to_numpy() >= ventral_conf_thresh
            is_valid |= (finite & conf)

    segments = []
    for val, group in groupby(enumerate(is_valid), key=lambda x: x[1]):
        if val:
            items = list(group)
            s_idx = items[0][0]
            e_idx = items[-1][0] + 1
            if (e_idx - s_idx) >= MIN_SEGMENT_FRAMES:
                segments.append((s_idx, e_idx))
    return segments


def compute_median_duration(states_z):
    """Computes median bout duration in frames and active syllable count from discrete states."""
    durations = []
    active_sylls = set()
    for seq in np.asarray(states_z):
        for val, grp in groupby(seq):
            if val >= 0:
                durations.append(len(list(grp)))
                active_sylls.add(int(val))
    med_fr = float(np.median(durations)) if durations else float("nan")
    med_ms = (med_fr / TARGET_FPS) * 1000.0 if np.isfinite(med_fr) else float("nan")
    return med_fr, med_ms, len(active_sylls), len(durations)


def compute_consecutive_ari_convergence(checkpoint_path):
    """Computes pairwise Adjusted Rand Index across all saved checkpoint snapshots."""
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
        from sklearn.metrics import adjusted_rand_score
        for i in range(len(avail_snaps) - 1):
            s_prev = avail_snaps[i]
            s_curr = avail_snaps[i + 1]
            z_prev = z_seqs[s_prev]
            z_curr = z_seqs[s_curr]
            valid = (z_prev >= 0) & (z_curr >= 0)
            if np.sum(valid) > 0:
                ari = float(adjusted_rand_score(z_prev[valid], z_curr[valid]))
                flips = float(np.mean(z_prev[valid] != z_curr[valid]) * 100.0)
                ari_results[f"iter_{s_prev}_to_{s_curr}"] = {
                    "ari": ari,
                    "label_change_pct": flips
                }
        return ari_results
    except Exception as e:
        print(f"  [ARI Warning] Failed to compute ARI: {e}")
        return {}


def compute_dynamic_training_schedule(
    num_ar_iters: int,
    num_slds_iters: int,
    chunk_size: int = 50,
    current_iter: int = 0
):
    """Dynamically builds execution chunks up to num_ar_iters + num_slds_iters."""
    total_iters = num_ar_iters + num_slds_iters
    if current_iter >= total_iters:
        return []

    start = current_iter
    chunks = []
    while start < total_iters:
        end = min(start + chunk_size, total_iters)
        ar_only = (end <= num_ar_iters)
        phase_name = f"AR Phase ({start} -> {end})" if ar_only else f"SLDS Phase ({start} -> {end})"
        chunks.append((start, end, ar_only, phase_name))
        start = end
    return chunks


def ingest_or_load_data(config):
    """Ingests all 118 static fused sessions or loads cached preprocessed data."""
    preprocessed_cache = COHORT_DIR / "preprocessed_data.pkl"
    vfi_path = COHORT_DIR / "video_frame_indexes.npz"

    if preprocessed_cache.exists() and vfi_path.exists():
        print(f"  [CACHE HIT] Loading preprocessed data from {preprocessed_cache}...", flush=True)
        with open(preprocessed_cache, "rb") as f:
            data, metadata = pickle.load(f)
        with np.load(vfi_path) as z:
            video_frame_indexes = {k: z[k] for k in z.files}
        return data, metadata, video_frame_indexes

    print(f"  [INGESTING] Ingesting 118 epoch-corrected static fused sessions from {DATA_SOURCE_DIR}...", flush=True)
    h5_files = sorted(DATA_SOURCE_DIR.glob("*_tandem_21kp.h5"))
    print(f"  Discovered {len(h5_files)} HDF5 files to ingest.")

    if len(h5_files) != 118:
        raise RuntimeError(f"Expected 118 input files in {DATA_SOURCE_DIR}, found {len(h5_files)}!")

    split_coords = {}
    split_confs = {}
    video_frame_indexes = {}
    total_retained = 0

    t0 = time.time()
    for i, f_path in enumerate(h5_files, 1):
        sid = f_path.name.replace("_tandem_21kp.h5", "")
        df = pd.read_hdf(f_path)
        scorer = df.columns.levels[0][0]
        segments = split_dual_view_segments(df)

        coords_arr = np.concatenate([
            np.column_stack([df[(scorer, bp, "x")].to_numpy() for bp in BODYPARTS_21KP])[:, :, None],
            np.column_stack([df[(scorer, bp, "y")].to_numpy() for bp in BODYPARTS_21KP])[:, :, None]
        ], axis=-1).astype(np.float32)

        confs_arr = np.column_stack([df[(scorer, bp, "likelihood")].to_numpy() for bp in BODYPARTS_21KP]).astype(np.float32)

        for seg_i, (s_idx, e_idx) in enumerate(segments):
            seg_name = f"{sid}_seg{seg_i:03d}"
            split_coords[seg_name] = coords_arr[s_idx:e_idx].copy()
            split_confs[seg_name] = confs_arr[s_idx:e_idx].copy()
            video_frame_indexes[seg_name] = np.arange(s_idx, e_idx, dtype=np.int32)
            total_retained += (e_idx - s_idx)

        if i % 25 == 0 or i == len(h5_files):
            print(f"    [{i:3d}/{len(h5_files)}] Processed {sid} | Total segments so far: {len(split_coords)} ({total_retained:,} frames)", flush=True)

    print(f"\n  Ingestion complete in {time.time() - t0:.1f}s. Formatting data in Host RAM...", flush=True)
    np.savez_compressed(vfi_path, **video_frame_indexes)

    with jax.default_device(jax.devices("cpu")[0]):
        data, metadata = kpms.format_data(split_coords, split_confs, **config)

    with open(preprocessed_cache, "wb") as f:
        pickle.dump((data, metadata), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached preprocessed dataset to {preprocessed_cache} ({len(data['Y'])} segments, {total_retained:,} frames)", flush=True)

    write_manifest(
        preprocessed_cache,
        kind="kpms_preprocessed_cohort_v2",
        inputs=h5_files,
        config=config,
        extra={"total_segments": len(split_coords), "total_frames": total_retained}
    )

    return data, metadata, video_frame_indexes


def run_single_seed(seed, seed_idx, total_seeds, data, metadata, pca, config, args):
    """Executes a single clean production run with 50-iteration ntfy reporting."""
    total_iters = args.num_ar_iters + args.num_slds_iters
    model_name = f"model_21kp_static_{args.num_ar_iters}ar_{args.num_slds_iters}slds_seed{seed}"
    project_dir = COHORT_DIR / model_name
    project_dir.mkdir(parents=True, exist_ok=True)

    final_results = project_dir / "results.h5"
    if final_results.exists():
        print(f"\n✅ Seed {seed} already has completed results at {final_results}. Skipping!", flush=True)
        return

    print("\n" + "=" * 80)
    print(f"🚀 [SEED {seed_idx}/{total_seeds}] STARTING PRODUCTION TRAINING: SEED {seed} (V2)")
    print(f"  Model Name:       {model_name}")
    print(f"  Regimen:          {args.num_ar_iters} AR + {args.num_slds_iters} SLDS = {total_iters} TOTAL ITERS")
    print(f"  Kappa:            Init {args.init_kappa:.2e} -> SLDS {args.slds_kappa:.2e}")
    print(f"  Save Checkpoints: Every {args.save_every_n} iterations")
    print(f"  Directory:        {project_dir}")
    print("=" * 80, flush=True)

    send_ntfy(
        f"Seed {seed} Started ({seed_idx}/{total_seeds}) ⏳",
        f"Model: {model_name}\n"
        f"Regimen: {args.num_ar_iters} AR + {args.num_slds_iters} SLDS ({total_iters} total iters)\n"
        f"118 Epoch-Corrected Sessions (~23.8M frames) on RTX 4090\n"
        f"SLDS Kappa: {args.slds_kappa:.2e}",
        priority="default",
        tags="hourglass_flowing_sand,seedling"
    )

    t_seed_start = time.time()

    # Apply monkeypatch and Amdahl optimizations
    apply_kpms_checkpoint_patch()
    rb.apply_optimizations()

    # Initialize model from scratch
    print(f"\nInitializing model from scratch (Seed {seed})...", flush=True)
    model = batched_init_model_by_video(data, pca, config, seed=seed)

    # Precompute GPU batch config
    print("Precomputing optimal batch configuration on GPU...", flush=True)
    rb.KPMSStateCache.clear()
    rb.precompute_optimal_batch_config(data, model=model, verbose=False)

    # Dynamic 50-Iteration Chunks
    chunks = compute_dynamic_training_schedule(
        args.num_ar_iters,
        args.num_slds_iters,
        chunk_size=50,
        current_iter=0
    )

    print(f"\nScheduled {len(chunks)} training chunks:")
    for c_i, (cs, ce, car, cdesc) in enumerate(chunks, 1):
        print(f"  Chunk {c_i}: {cdesc} [Iter {cs} -> {ce}] (AR-only: {car})")

    for (c_start, c_end, c_ar_only, c_desc) in chunks:
        print(f"\n{'=' * 60}")
        print(f"▶️ RUNNING CHUNK: {c_desc} [Iter {c_start} -> {c_end} of {total_iters}]")
        print(f"{'=' * 60}", flush=True)

        if not c_ar_only and c_start == args.num_ar_iters:
            print(f">>> Transitioning to SLDS Phase (kappa = {args.slds_kappa:.2e})...", flush=True)
            model = kpms.update_hypparams(model, kappa=float(args.slds_kappa))
            rb.KPMSStateCache.clear()
            rb.precompute_optimal_batch_config(data, model=model, verbose=False)

        t_chunk_start = time.time()
        model, _ = kpms.fit_model(
            model, data, metadata,
            project_dir=str(COHORT_DIR),
            model_name=model_name,
            ar_only=c_ar_only,
            start_iter=c_start,
            num_iters=c_end,
            parallel_message_passing=False,
            save_every_n_iters=args.save_every_n,
            generate_progress_plots=False,
            verbose=True,
        )
        chunk_elapsed_min = (time.time() - t_chunk_start) / 60.0
        total_elapsed_min = (time.time() - t_seed_start) / 60.0

        # Calculate current telemetry from states z
        med_fr, med_ms, n_sylls, n_bouts = compute_median_duration(model["states"]["z"])

        # Calculate ETA
        iters_done = c_end
        iters_left = total_iters - iters_done
        rate_min_per_iter = total_elapsed_min / max(1, iters_done)
        eta_min = iters_left * rate_min_per_iter

        print(f"\n🔔 CHUNK {c_end}/{total_iters} COMPLETE in {chunk_elapsed_min:.2f} min.")
        print(f"   Current Median:  {med_fr:.1f} frames ({med_ms:.1f} ms)")
        print(f"   Active Sylls:    {n_sylls} (over {n_bouts:,} bouts)")
        print(f"   Elapsed:         {total_elapsed_min:.1f} min | ETA: {eta_min:.1f} min", flush=True)

        # Send ntfy alert for each 50 iters
        is_final_chunk = (c_end == total_iters)
        ntfy_title = f"V2 Seed {seed} Iter {c_end}/{total_iters} {'COMPLETE! 🎉' if is_final_chunk else 'Reached ⏱️'}"
        ntfy_body = (
            f"Seed: {seed} ({seed_idx}/{total_seeds})\n"
            f"Progress: {c_end}/{total_iters} iters ({c_desc})\n"
            f"Median Duration: {med_fr:.1f} fr ({med_ms:.1f} ms)\n"
            f"Active Syllables: {n_sylls}\n"
            f"Elapsed: {total_elapsed_min:.1f} min | ETA: {eta_min:.1f} min"
        )
        send_ntfy(ntfy_title, ntfy_body, priority="high" if is_final_chunk else "default", tags="tada" if is_final_chunk else "bell")

        gc.collect()

    seed_total_min = (time.time() - t_seed_start) / 60.0
    print(f"\n[SEED {seed} FINISHED] Total training time: {seed_total_min:.2f} minutes.", flush=True)

    # Extract final results
    print("  Extracting final results...", flush=True)
    results = kpms.extract_results(model, metadata, str(COHORT_DIR), model_name, save_results=True)
    print(f"  Saved final results to {final_results}")

    # Evaluate Convergence ARI
    chk_path = project_dir / "checkpoint.h5"
    if chk_path.exists():
        print("  Evaluating consecutive-iteration ARI convergence...", flush=True)
        ari_convergence = compute_consecutive_ari_convergence(chk_path)
        for step, m in ari_convergence.items():
            print(f"    {step}: ARI = {m['ari']:.4f} | Label Flip = {m['label_change_pct']:.1f}%")
        conv_file = project_dir / "ari_convergence.json"
        conv_file.write_text(json.dumps(ari_convergence, indent=2))
        print(f"  Saved convergence report to {conv_file}")

    send_ntfy(
        f"V2 Seed {seed} Finished ({seed_idx}/{total_seeds}) ✅",
        f"Model {model_name} completed in {seed_total_min:.1f} min.\n"
        f"Final Median: {med_fr:.1f} fr ({med_ms:.1f} ms) across {n_sylls} syllables.\n"
        f"Results saved to results.h5.",
        priority="high",
        tags="white_check_mark,tada"
    )


def main():
    parser = argparse.ArgumentParser(description="Full 118-session cohort 21kp static production queue (V2).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101],
                        help="List of seeds to execute sequentially (default: 101)")
    parser.add_argument("--num-ar-iters", type=int, default=50,
                        help="Number of AR iterations (default: 50)")
    parser.add_argument("--num-slds-iters", type=int, default=300,
                        help="Number of SLDS iterations (default: 300)")
    parser.add_argument("--init-kappa", type=float, default=1.0e6,
                        help="Initial AR kappa (default: 1.0e6)")
    parser.add_argument("--slds-kappa", type=float, default=1.5e15,
                        help="Calibrated cohort SLDS kappa (default: 1.5e15)")
    parser.add_argument("--save-every-n", type=int, default=25,
                        help="Checkpoint save interval (default: 25)")
    args = parser.parse_args()

    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    total_iters = args.num_ar_iters + args.num_slds_iters

    print("=" * 80)
    print("🚀 LAUNCHING CLEAN FULL COHORT STATIC 21KP V2 PRODUCTION RUN")
    print(f"  Cohort Directory:    {COHORT_DIR}")
    print(f"  Data Source:         {DATA_SOURCE_DIR}")
    print(f"  Seeds to execute:    {args.seeds}")
    print(f"  Regimen per seed:    {args.num_ar_iters} AR + {args.num_slds_iters} SLDS = {total_iters} iters")
    print(f"  SLDS Kappa:          {args.slds_kappa:.2e} (Calibrated 1.5x scale)")
    print(f"  ntfy.sh Endpoint:    {NTFY_URL} (Alerts sent every 50 iterations)")
    print("=" * 80, flush=True)

    # 1. Config setup
    config_path = COHORT_DIR / "config.yml"
    if not config_path.exists():
        src_cfg = EXPERIMENT_DIR / "cohort_static_21kp/config.yml"
        if not src_cfg.exists():
            src_cfg = EXPERIMENT_DIR / "pilot_comparison_21kp_static/config.yml"
        shutil.copy(src_cfg, config_path)

    config = kpms.load_config(str(COHORT_DIR))
    config["fps"] = TARGET_FPS
    config["seg_length"] = 10030
    config["bodyparts"] = BODYPARTS_21KP
    config["use_bodyparts"] = BODYPARTS_21KP
    if "skeleton" in config and config["skeleton"]:
        config["skeleton"] = [
            edge for edge in config["skeleton"]
            if edge[0] in BODYPARTS_21KP and edge[1] in BODYPARTS_21KP
        ]
    config["anterior_bodyparts"] = ["tail_base"]
    config["posterior_bodyparts"] = ["nose"]
    config["trans_hypparams"]["kappa"] = args.init_kappa
    kpms.update_config(str(COHORT_DIR), **config)
    config = kpms.load_config(str(COHORT_DIR))

    # 2. Ingest or load cached data
    data, metadata, video_frame_indexes = ingest_or_load_data(config)

    # 3. Fit or load PCA
    pca_file = COHORT_DIR / "pca.p"
    if pca_file.exists():
        print(f"  [CACHE HIT] Loading fitted PCA from {pca_file}...", flush=True)
        pca = kpms.load_pca(str(COHORT_DIR))
    else:
        print("  Fitting PCA on representative subset of cohort...", flush=True)
        pca_subset_n = min(50, len(data["Y"]))
        subset_idx = np.linspace(0, len(data["Y"]) - 1, pca_subset_n, dtype=int)
        data_pca_subset = {
            k: v[subset_idx] if (hasattr(v, "__len__") and len(v) == len(data["Y"])) else v
            for k, v in data.items()
        }
        pca = kpms.fit_pca(**data_pca_subset, **config)
        with open(pca_file, "wb") as f:
            pickle.dump(pca, f)
        print(f"  Fitted and cached PCA to {pca_file}", flush=True)

        write_manifest(
            pca_file,
            kind="kpms_pca_cohort_v2",
            inputs=[COHORT_DIR / "preprocessed_data.pkl"],
            config=config,
            extra={"pca_dim": pca.components_.shape[0]}
        )

    # 4. Sequential Queue Execution
    send_ntfy(
        "KPMS V2 Cohort Retraining Launched 🚀",
        f"Seed: {args.seeds}\n"
        f"Plan: {args.num_ar_iters} AR + {args.num_slds_iters} SLDS = {total_iters} iters\n"
        f"118 Epoch-Corrected Sessions (~23.8M frames @ 60 FPS)\n"
        f"Kappa: {args.slds_kappa:.2e}\n"
        f"Alerts will be pushed every 50 iterations.",
        priority="high",
        tags="rocket,brain"
    )

    t_queue_start = time.time()
    for idx, seed in enumerate(args.seeds, 1):
        run_single_seed(seed, idx, len(args.seeds), data, metadata, pca, config, args)

    total_queue_hours = (time.time() - t_queue_start) / 3600.0
    print("\n" + "=" * 80)
    print(f"🎉 PRODUCTION RUN V2 COMPLETED in {total_queue_hours:.2f} hours!")
    print("=" * 80, flush=True)

    send_ntfy(
        "V2 Production Run Complete! 🏆",
        f"Completed seed {args.seeds} in {total_queue_hours:.2f} hours.\n"
        f"All models, checkpoints, results, and ARI reports saved in {COHORT_DIR}.",
        priority="high",
        tags="trophy,tada,partying_face"
    )


if __name__ == "__main__":
    main()
