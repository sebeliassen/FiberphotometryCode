#!/usr/bin/env python3
"""
run_pilot_17kp_vs_21kp_comparison.py
====================================
Controlled 10-session benchmark comparing:
  1. arm_17kp: 17 Dorsal keypoints (TopCam baseline, 60fps)
  2. arm_21kp_static: 17 Dorsal + 4 Paws (Static Affine registration)
  3. arm_21kp_dynamic: 17 Dorsal + 4 Paws (Dynamic re-anchoring, omitting bot anchors)

Training regimen:
  - 50 AR + 150 SLDS (200 total iterations)
  - Saves intermediate checkpoints every 25 iterations during SLDS.
  - Automatically measures consecutive-iteration Adjusted Rand Index (ARI)
    to quantify syllable landscape convergence / stability.
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from itertools import groupby
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Unbuffered live stdout
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
MIN_SEGMENT_FRAMES = 180

DORSAL_BPS = [
    'nose', 'left_ear_tip', 'right_ear_tip', 'neck',
    'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3',
    'tail_base', 'left_shoulder', 'left_midside', 'left_hip',
    'right_shoulder', 'right_midside', 'right_hip', 'head_midpoint'
]
PAW_BPS = ['left_front_paw', 'right_front_paw', 'left_hind_paw', 'right_hind_paw']

PILOT_SESSIONS = [
    '2026-03-05T11_01_01_top_left',
    '2026-03-05T12_46_18_top_left',
    '2026-03-06T09_48_02_top_left',
    '2026-03-06T11_14_44_top_left',
    '2026-03-05T11_01_01_bottom_left',
    '2026-03-05T12_46_18_bottom_left',
    '2026-03-06T09_48_02_bottom_left',
    '2026-03-05T11_01_01_bottom_right',
    '2026-03-05T12_46_18_bottom_right',
    '2026-03-06T09_48_02_bottom_right'
]


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


def run_arm(
    arm: str,
    num_ar_iters: int = 50,
    num_slds_iters: int = 150,
    init_kappa: float = 1.0e6,
    slds_kappa: float = 1.5e12,
    seed: int = 404,
    save_every_n: int = 25,
    model_suffix: str = "",
):
    total_iters = num_ar_iters + num_slds_iters
    print("=" * 80)
    print(f"🚀 LAUNCHING CONTROLLED PILOT BENCHMARK ARM: {arm.upper()}")
    print(f"  AR Iterations:       {num_ar_iters}")
    print(f"  SLDS Iterations:     {num_slds_iters} (Total: {total_iters})")
    print(f"  Init Kappa:          {init_kappa:.2e} -> SLDS Kappa: {slds_kappa:.2e}")
    print(f"  Save Checkpoint:     Every {save_every_n} iterations")
    print(f"  Seed:                {seed}")
    print("=" * 80, flush=True)

    arm_dir = EXP_DIR / f"pilot_comparison_{arm}"
    model_name = f"model_{arm}_{model_suffix}_seed{seed}" if model_suffix else f"model_{arm}_seed{seed}"
    project_dir = arm_dir / model_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous run files to avoid h5py append/key collision
    for old_f in [project_dir / "results.h5", project_dir / "checkpoint.h5", project_dir / "ari_convergence.json"]:
        if old_f.exists():
            try:
                old_f.unlink()
            except Exception:
                pass

    if arm == "17kp":
        use_bodyparts = DORSAL_BPS
        data_source_dir = EXP_DIR / "data/pilot_fused_static_21kp"
        file_suffix = "_tandem_21kp.h5"
    elif arm == "21kp_static":
        use_bodyparts = DORSAL_BPS + PAW_BPS
        data_source_dir = EXP_DIR / "data/pilot_fused_static_21kp"
        file_suffix = "_tandem_21kp.h5"
    elif arm == "21kp_dynamic":
        use_bodyparts = DORSAL_BPS + PAW_BPS
        data_source_dir = EXP_DIR / "data/fused_h5_continuous_calibrated"
        file_suffix = "_tandem_23kp.h5"
    else:
        raise ValueError(f"Unknown arm '{arm}'")

    print(f"Configuring {len(use_bodyparts)} bodyparts: {use_bodyparts}")
    print(f"Data source: {data_source_dir}\n")

    # 1. Config setup
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

    # 2. Ingest data
    preprocessed_cache = arm_dir / "preprocessed_data.pkl"
    vfi_path = arm_dir / "video_frame_indexes.npz"

    if preprocessed_cache.exists() and vfi_path.exists():
        print(f"  [CACHE HIT] Loading preprocessed data from {preprocessed_cache}...", flush=True)
        with open(preprocessed_cache, "rb") as f:
            data, metadata = pickle.load(f)
        with np.load(vfi_path) as z:
            video_frame_indexes = {k: z[k] for k in z.files}
    else:
        print(f"  [INGESTING] Ingesting files for {arm}...", flush=True)
        split_coords = {}
        split_confs = {}
        video_frame_indexes = {}
        total_retained = 0

        for sid in PILOT_SESSIONS:
            f_path = data_source_dir / f"{sid}{file_suffix}"
            if not f_path.exists():
                print(f"  ⚠️ Warning: {f_path} does not exist, skipping.")
                continue

            df = pd.read_hdf(f_path)
            scorer = df.columns.levels[0][0]
            segments = split_dual_view_segments(df)

            coords_arr = np.concatenate([
                np.column_stack([df[(scorer, bp, "x")].to_numpy() for bp in use_bodyparts])[:, :, None],
                np.column_stack([df[(scorer, bp, "y")].to_numpy() for bp in use_bodyparts])[:, :, None]
            ], axis=-1).astype(np.float32)

            confs_arr = np.column_stack([df[(scorer, bp, "likelihood")].to_numpy() for bp in use_bodyparts]).astype(np.float32)

            for seg_i, (s_idx, e_idx) in enumerate(segments):
                seg_name = f"{sid}_seg{seg_i:03d}"
                split_coords[seg_name] = coords_arr[s_idx:e_idx].copy()
                split_confs[seg_name] = confs_arr[s_idx:e_idx].copy()
                video_frame_indexes[seg_name] = np.arange(s_idx, e_idx, dtype=np.int32)
                total_retained += (e_idx - s_idx)

        print(f"  Formatted {len(split_coords)} segments with {total_retained:,} frames.")
        np.savez_compressed(vfi_path, **video_frame_indexes)

        with jax.default_device(jax.devices("cpu")[0]):
            data, metadata = kpms.format_data(split_coords, split_confs, **config)

        with open(preprocessed_cache, "wb") as f:
            pickle.dump((data, metadata), f, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. Fit PCA
    pca_file = arm_dir / "pca.p"
    if pca_file.exists():
        pca = kpms.load_pca(str(arm_dir))
    else:
        print("  Fitting PCA on representative subset...", flush=True)
        pca_subset_n = min(30, len(data["Y"]))
        subset_idx = np.linspace(0, len(data["Y"]) - 1, pca_subset_n, dtype=int)
        data_pca_subset = {
            k: v[subset_idx] if (hasattr(v, "__len__") and len(v) == len(data["Y"])) else v
            for k, v in data.items()
        }
        pca = kpms.fit_pca(**data_pca_subset, **config)
        with open(pca_file, "wb") as f:
            pickle.dump(pca, f)

    # 4. Initialize & Optimize
    print("  Precomputing optimal batch configuration on GPU...", flush=True)
    rb.KPMSStateCache.clear()
    rb.precompute_optimal_batch_config(data, verbose=False)

    print(f"  Initializing model (Seed {seed})...", flush=True)
    model = batched_init_model_by_video(data, pca, config, seed=seed)
    model = kpms.update_hypparams(model, kappa=float(init_kappa))

    apply_kpms_checkpoint_patch()
    rb.apply_optimizations()

    # 5. Gibbs Training
    t_train_start = time.time()
    if num_ar_iters > 0:
        print(f"\n>>> Running AR Phase (iters 0 -> {num_ar_iters})...", flush=True)
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

    # Transition to SLDS kappa
    print(f"\n>>> Transitioning to SLDS Phase (kappa = {slds_kappa:.2e})...", flush=True)
    model = kpms.update_hypparams(model, kappa=float(slds_kappa))
    rb.KPMSStateCache.clear()
    rb.precompute_optimal_batch_config(data, model=model, verbose=False)

    if num_slds_iters > 0:
        print(f"\n>>> Running SLDS Phase (iters {num_ar_iters} -> {total_iters})...", flush=True)
        model, _ = kpms.fit_model(
            model, data, metadata,
            project_dir=str(arm_dir),
            model_name=model_name,
            ar_only=False,
            start_iter=num_ar_iters,
            num_iters=total_iters,
            parallel_message_passing=False,
            save_every_n_iters=save_every_n,
            generate_progress_plots=False,
            verbose=True,
        )

    train_time_min = (time.time() - t_train_start) / 60.0
    print(f"\n[TRAINING FINISHED] Total time: {train_time_min:.2f} minutes.", flush=True)

    # 6. Extract results
    print("  Extracting results...", flush=True)
    res_path = project_dir / "results.h5"
    if res_path.exists():
        try:
            res_path.unlink()
        except Exception:
            pass
    results = kpms.extract_results(model, metadata, str(arm_dir), model_name, save_results=True)
    print(f"  Saved results to {res_path}")

    # 7. Evaluate Convergence ARI across checkpoints
    chk_path = project_dir / "checkpoint.h5"
    if chk_path.exists():
        print("\n  Evaluating consecutive-iteration ARI convergence...", flush=True)
        ari_convergence = compute_consecutive_ari_convergence(chk_path)
        for step, m in ari_convergence.items():
            print(f"    {step}: ARI = {m['ari']:.4f} | Label Flip = {m['label_change_pct']:.1f}%")
        conv_file = project_dir / "ari_convergence.json"
        conv_file.write_text(json.dumps(ari_convergence, indent=2))
        print(f"  Saved convergence report to {conv_file}")

    return {
        "arm": arm,
        "runtime_min": train_time_min,
        "results_path": str(project_dir / "results.h5"),
    }


def main():
    parser = argparse.ArgumentParser(description="Controlled 17kp vs 21kp pilot comparison runner.")
    parser.add_argument("--arms", nargs="+", choices=["17kp", "21kp_static", "21kp_dynamic", "all"], default=["17kp", "21kp_static"],
                        help="Arms to run (default: 17kp 21kp_static)")
    parser.add_argument("--num-ar-iters", type=int, default=50, help="Number of AR iterations")
    parser.add_argument("--num-slds-iters", type=int, default=150, help="Number of SLDS iterations")
    parser.add_argument("--init-kappa", type=float, default=1.0e6, help="Initial AR kappa")
    parser.add_argument("--slds-kappa", type=float, default=1.5e12, help="SLDS kappa")
    parser.add_argument("--save-every-n", type=int, default=25, help="Save interval for snapshots")
    parser.add_argument("--seed", type=int, default=404, help="Random seed")
    parser.add_argument("--model-suffix", type=str, default="", help="Model suffix")
    args = parser.parse_args()

    arms = ["17kp", "21kp_static", "21kp_dynamic"] if "all" in args.arms else args.arms
    print(f"Will execute arms: {arms}")

    summary = {}
    for arm in arms:
        res = run_arm(
            arm=arm,
            num_ar_iters=args.num_ar_iters,
            num_slds_iters=args.num_slds_iters,
            init_kappa=args.init_kappa,
            slds_kappa=args.slds_kappa,
            seed=args.seed,
            save_every_n=args.save_every_n,
            model_suffix=args.model_suffix,
        )
        summary[arm] = res

    print("\n" + "=" * 80)
    print("🎉 ALL REQUESTED ARMS COMPLETED!")
    for arm, res in summary.items():
        print(f"  {arm:15s}: {res['runtime_min']:.2f} min -> {res['results_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
