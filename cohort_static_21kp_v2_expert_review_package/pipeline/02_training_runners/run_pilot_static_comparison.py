#!/usr/bin/env python3
"""
run_pilot_static_comparison.py
==============================
High-performance pilot comparison runner between:
  - Arm A (21kp): 17 Dorsal + 4 Ventral Paws (Static Affine Mapped)
  - Arm B (23kp): 17 Dorsal + 4 Ventral Paws + 2 Bottom Anchors (Static Affine Mapped)

Integrates:
  1. Amdahl 27x Banded Cholesky GMRF solver (utils.resample_batching)
  2. Preemptive analytical batch sizing (rb.precompute_optimal_batch_config)
  3. Bounded rolling checkpoint compactor (compact_checkpoint & patch)
  4. Host DRAM staging (jax.default_device(cpu))
  5. Empirically validated kappa (init: 1e6 -> SLDS: 1e5)
"""

import os
import sys

# Live unbuffered stdout/stderr
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Headless environment guards: prevent plotly/webbrowser hangs
os.environ["BROWSER"] = "/bin/true"
import webbrowser
webbrowser.get = lambda *args, **kwargs: None

# Hardware environment flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["USE_MEMORY_PATCH"] = "1"

import time
import argparse
import shutil
import pickle
import gc
from pathlib import Path
from itertools import groupby
import numpy as np
import pandas as pd
import h5py
import psutil

# Patch numpy 2.x bool8 if needed before importing kpms
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import matplotlib
matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import keypoint_moseq as kpms

# Setup repository paths
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
from syllable_generation.syllabizers.keypoint_moseq_syll.utils.compact_checkpoint import compact_checkpoint

from core.camera_correspondence import get_session_mapping
from jax_moseq.models.keypoint_slds.alignment import center_embedding

# Constants
TARGET_FPS = 60
MIN_GAP_FRAMES = 180
MIN_SEGMENT_FRAMES = 180

DORSAL_BPS = [
    'nose', 'left_ear_tip', 'right_ear_tip', 'neck',
    'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3',
    'tail_base', 'left_shoulder', 'left_midside', 'left_hip',
    'right_shoulder', 'right_midside', 'right_hip', 'head_midpoint'
]
PAW_BPS = ['left_front_paw', 'right_front_paw', 'left_hind_paw', 'right_hind_paw']
ANCHOR_BPS = ['bot_nose_tip', 'bot_tail_base']


def split_session_into_dual_view_segments(df: pd.DataFrame, ventral_conf_thresh: float = 0.4):
    """Identifies continuous dual-view segments excluding prolonged missing ventral gaps."""
    scorer = df.columns.levels[0][0]
    n_frames = len(df)
    is_valid = np.zeros(n_frames, dtype=bool)

    # Ventral keypoints must be present
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
    return segments, is_valid


def run_arm(
    arm: str,
    num_ar_iters: int = 50,
    num_slds_iters: int = 100,
    init_kappa: float = 1.0e6,
    decrease_kappa_factor: float = 10.0,
    seed: int = 404,
    save_every_n: int = 25,
    model_suffix: str = "",
):
    """Executes a single pilot training arm."""
    slds_kappa = init_kappa / decrease_kappa_factor
    total_iters = num_ar_iters + num_slds_iters

    print("=" * 80)
    print(f"🚀 EXECUTING PILOT TRAINING ARM: {arm.upper()}")
    print(f"  AR Iterations:       {num_ar_iters}")
    print(f"  SLDS Iterations:     {num_slds_iters} (Total: {total_iters})")
    print(f"  Init Kappa:          {init_kappa:.2e} -> SLDS Kappa: {slds_kappa:.2e}")
    print(f"  Seed:                {seed}")
    if model_suffix:
        print(f"  Model Suffix:        {model_suffix}")
    print("=" * 80, flush=True)

    arm_dir = EXP_DIR / f"pilot_static_{arm}"
    fused_dir = EXP_DIR / f"data/pilot_fused_static_{arm}"
    model_name = f"pilot_static_{arm}_{model_suffix}_seed{seed}" if model_suffix else f"pilot_static_{arm}_seed{seed}"
    project_dir = arm_dir / model_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    use_bodyparts = DORSAL_BPS + PAW_BPS + (ANCHOR_BPS if arm == "23kp" else [])
    print(f"Configuring {len(use_bodyparts)} bodyparts: {use_bodyparts}\n")

    # 1. Setup config
    config_source = EXP_DIR / "kpms_tandem_cohort_production/config.yml"
    config_dest = arm_dir / "config.yml"
    if not config_dest.exists():
        shutil.copy(config_source, config_dest)

    config = kpms.load_config(str(arm_dir))
    config["fps"] = TARGET_FPS
    config["seg_length"] = 10030
    config["bodyparts"] = use_bodyparts
    config["use_bodyparts"] = use_bodyparts
    if "skeleton" in config and config["skeleton"]:
        config["skeleton"] = [
            edge for edge in config["skeleton"]
            if edge[0] in use_bodyparts and edge[1] in use_bodyparts
        ]
    config["anterior_bodyparts"] = ["tail_base"]
    config["posterior_bodyparts"] = ["nose"]
    config["trans_hypparams"]["kappa"] = init_kappa
    kpms.update_config(str(arm_dir), **config)
    config = kpms.load_config(str(arm_dir))

    # 2. Ingest and format data in Host DRAM
    preprocessed_cache = arm_dir / "preprocessed_data.pkl"
    vfi_path = arm_dir / "video_frame_indexes.npz"

    if preprocessed_cache.exists() and vfi_path.exists():
        print(f"  [CACHE HIT] Loading preprocessed data from {preprocessed_cache}...", flush=True)
        with open(preprocessed_cache, "rb") as f:
            data, metadata = pickle.load(f)
        with np.load(vfi_path) as z:
            video_frame_indexes = {k: z[k] for k in z.files}
    else:
        print(f"  [INGESTING] Ingesting fused static H5 files from {fused_dir}...", flush=True)
        fused_files = sorted(fused_dir.glob(f"*_tandem_{arm}.h5"))
        print(f"  Found {len(fused_files)} files to ingest.")

        split_coords = {}
        split_confs = {}
        video_frame_indexes = {}
        total_raw = 0
        total_retained = 0

        for f_idx, f_path in enumerate(fused_files):
            sid = f_path.name.replace(f"_tandem_{arm}.h5", "")
            df = pd.read_hdf(f_path)
            N = len(df)
            total_raw += N
            scorer = df.columns.levels[0][0]

            segments, _ = split_session_into_dual_view_segments(df)
            retained_here = sum(e - s for s, e in segments)
            total_retained += retained_here

            coords_arr = np.concatenate([
                np.column_stack([df[(scorer, bp, "x")].to_numpy() for bp in use_bodyparts])[:, :, None],
                np.column_stack([df[(scorer, bp, "y")].to_numpy() for bp in use_bodyparts])[:, :, None]
            ], axis=-1).astype(np.float32)

            confs_arr = np.column_stack([df[(scorer, bp, "likelihood")].to_numpy() for bp in use_bodyparts]).astype(np.float32)

            for seg_i, (s_idx, e_idx) in enumerate(segments):
                seg_name = f"{sid}_seg{seg_i:03d}"
                split_coords[seg_name] = coords_arr[s_idx:e_idx].copy()
                split_confs[seg_name] = confs_arr[s_idx:e_idx].copy()
                video_frame_indexes[seg_name] = np.arange(s_idx, e_idx, dtype=np.int64)
            del coords_arr, confs_arr, df

        np.savez_compressed(vfi_path, **video_frame_indexes)

        print(f"  Formatting data with Keypoint-MoSeq across {len(split_coords)} segments in Host DRAM...", flush=True)
        cpu_device = jax.devices("cpu")[0]
        with jax.default_device(cpu_device):
            data, metadata = kpms.format_data(split_coords, split_confs, **config)

        del split_coords, split_confs
        gc.collect()

        data = {
            "Y": np.asarray(data["Y"], dtype=np.float64),
            "conf": np.asarray(data["conf"], dtype=np.float32),
            "mask": np.asarray(data["mask"], dtype=np.float32),
        }

        with open(preprocessed_cache, "wb") as f:
            pickle.dump((data, metadata), f, protocol=pickle.HIGHEST_PROTOCOL)

    total_frames = int(data["mask"].sum())
    print(f"  Active frames: {total_frames:,} across {data['Y'].shape[0]} segments.")

    # 3. Fit PCA and inspect keypoint variance distribution
    pca_p = arm_dir / "pca.p"
    if pca_p.exists():
        pca = kpms.load_pca(str(arm_dir))
    else:
        print("\n  Fitting PCA on representative subset (D=10)...", flush=True)
        pca_subset_n = min(30, len(data["Y"]))
        subset_idx = np.linspace(0, len(data["Y"]) - 1, pca_subset_n, dtype=int)
        data_pca_subset = {
            k: v[subset_idx] if (hasattr(v, "__len__") and len(v) == len(data["Y"])) else v
            for k, v in data.items()
        }
        pca = kpms.fit_pca(**data_pca_subset, **config)
        kpms.save_pca(pca, str(arm_dir))

    # Analyze PCA loadings by anatomical group
    K = len(use_bodyparts)
    G = np.array(center_embedding(K))
    C = pca.components_
    V = pca.explained_variance_ratio_
    tot_var = np.zeros(K)
    for c in range(10):
        w = C[c].reshape(K - 1, 2)
        kp_recon = G @ w
        e = (kp_recon**2).sum(axis=1)
        tot_var += (e / e.sum()) * V[c]
    tot_var /= tot_var.sum()

    dorsal_share = tot_var[:17].sum()
    paw_share = tot_var[17:21].sum()
    anchor_share = tot_var[21:].sum() if arm == "23kp" else 0.0

    print("\n" + "-" * 60)
    print(f"📊 PCA VARIANCE SHARE ANALYSIS ({arm.upper()}):")
    print(f"  Top 10 PCs Cumulative Variance: {V[:10].sum()*100:.2f}%")
    print(f"  17 Dorsal Keypoints Share:     {dorsal_share*100:.2f}%")
    print(f"  4 Ventral Paws Share:          {paw_share*100:.2f}%")
    if arm == "23kp":
        print(f"  2 Bottom Anchors Share:        {anchor_share*100:.2f}%")
    print("-" * 60 + "\n", flush=True)

    # 4. Preemptive optimal batching
    print("  Precomputing optimal batch configuration...", flush=True)
    rb.KPMSStateCache.clear()
    rb.precompute_optimal_batch_config(data, verbose=True)

    # 5. Initialize model
    print(f"  Initializing model with batched_init_model_by_video (Seed {seed})...", flush=True)
    model = batched_init_model_by_video(data, pca, config, seed=seed)
    model = kpms.update_hypparams(model, kappa=float(init_kappa))

    # 6. Apply rolling compactor & high-performance kernels
    print("  Enabling Bounded Rolling Checkpoint Compactor monkeypatch...", flush=True)
    apply_kpms_checkpoint_patch()
    print("  Enabling High-Performance MoSeq Kernels (Banded Cholesky + Batched GPU)...", flush=True)
    rb.apply_optimizations()

    checkpoint_file = project_dir / "checkpoint.h5"
    results_file = project_dir / "results.h5"

    # 7. Training Loop (Gibbs Iterations)
    print("\n" + "=" * 80)
    print(f"🔥 STARTING GIBBS TRAINING LOOP ({num_ar_iters} AR + {num_slds_iters} SLDS)")
    print("=" * 80, flush=True)

    t_train_start = time.time()

    # PHASE 1: AR Fitting
    if num_ar_iters > 0:
        print(f"\n>>> Running AR Phase: iters 0 -> {num_ar_iters}...", flush=True)
        model, _ = kpms.fit_model(
            model, data, metadata,
            project_dir=str(arm_dir),
            model_name=model_name,
            ar_only=True,
            start_iter=0,
            num_iters=num_ar_iters,
            parallel_message_passing=False,
            save_every_n_iters=save_every_n,
            generate_progress_plots=False,
            verbose=True,
        )
        print(f"[PHASE 1 COMPLETE] {num_ar_iters} AR iterations finished in {(time.time() - t_train_start)/60:.1f} min.", flush=True)

    # TRANSITION: Kappa reduction for SLDS
    print(f"\n  >>> TRANSITION TO SLDS PHASE (reducing kappa to {slds_kappa:.2e}) <<<", flush=True)
    model = kpms.update_hypparams(model, kappa=float(slds_kappa))
    rb.KPMSStateCache.clear()
    rb.precompute_optimal_batch_config(data, model=model, verbose=False)

    # PHASE 2: Full SLDS Fitting
    if num_slds_iters > 0:
        print(f"\n>>> Running SLDS Phase: iters {num_ar_iters} -> {total_iters}...", flush=True)
        t_slds_start = time.time()
        model, _ = kpms.fit_model(
            model, data, metadata,
            project_dir=str(arm_dir),
            model_name=model_name,
            ar_only=False,
            start_iter=num_ar_iters,
            num_iters=num_slds_iters,
            parallel_message_passing=False,
            save_every_n_iters=save_every_n,
            generate_progress_plots=False,
            verbose=True,
        )
        print(f"[PHASE 2 COMPLETE] {num_slds_iters} SLDS iterations finished in {(time.time() - t_slds_start)/60:.1f} min.", flush=True)

    print(f"\nTotal training completed in {(time.time() - t_train_start)/60:.2f} minutes.")

    # 8. Extract results
    print(f"  Extracting results for {model_name}...", flush=True)
    if results_file.exists():
        results_file.unlink()
    try:
        results = kpms.extract_results(model, metadata, str(arm_dir), model_name, save_results=True, overwrite=True)
    except TypeError:
        results = kpms.extract_results(model, metadata, str(arm_dir), model_name, save_results=True)
    print(f"  Saved results to {project_dir / 'results.h5'}. Done!")


def main():
    parser = argparse.ArgumentParser(description="Pilot static comparison runner.")
    parser.add_argument("--arm", type=str, default="both", choices=["21kp", "23kp", "both"], help="Training arm to run.")
    parser.add_argument("--num-ar-iters", type=int, default=50, help="Number of AR iterations.")
    parser.add_argument("--num-slds-iters", type=int, default=100, help="Number of SLDS iterations.")
    parser.add_argument("--init-kappa", type=float, default=1.0e6, help="Initial kappa for AR phase.")
    parser.add_argument("--decrease-kappa-factor", type=float, default=10.0, help="Kappa reduction factor.")
    parser.add_argument("--seed", type=int, default=404, help="Random seed.")
    parser.add_argument("--model-suffix", type=str, default="", help="Optional model name suffix.")
    args = parser.parse_args()

    arms_to_run = ["21kp", "23kp"] if args.arm == "both" else [args.arm]
    for arm in arms_to_run:
        run_arm(
            arm=arm,
            num_ar_iters=args.num_ar_iters,
            num_slds_iters=args.num_slds_iters,
            init_kappa=args.init_kappa,
            decrease_kappa_factor=args.decrease_kappa_factor,
            seed=args.seed,
            model_suffix=args.model_suffix,
        )

if __name__ == "__main__":
    main()
