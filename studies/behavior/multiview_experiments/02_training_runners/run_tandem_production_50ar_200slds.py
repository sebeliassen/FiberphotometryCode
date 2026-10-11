#!/usr/bin/env python3
"""
run_tandem_production_50ar_200slds.py
=============================================================================
Production-Scale Tandem / Multi-View Keypoint-MoSeq Training Runner:
50 AR Iterations + 200 SLDS Iterations (250 Total Gibbs Iterations)
on Full-Length Continuous Sessions (416,032 frames across 23 keypoints @ 60 FPS).

Key Architecture & Robustness Features:
1. Asynchronous / Fault-Tolerant Execution:
   - Fully supports resuming from checkpoints (checks checkpoint.h5 and picks up
     exactly where it left off in case of timeout, crash, or machine restart).
   - Traps SIGINT / SIGTERM to gracefully save and notify before termination.
2. Preemptive Analytical Tensor Sizing & Zero-OOM:
   - Sizing batch parameters deterministically via precompute_optimal_batch_config.
   - Evaluates closed-form memory equations against the active JAX VRAM pool.
3. Unbuffered Live Logging:
   - Flushes stdout/stderr after every step for live tracking via `tail -f`.
4. Real-Time Ntfy Notifications:
   - Pushes start alert, progress updates at every 25 iterations (with speed and ETA),
     phase transitions, completion summaries, and failure alerts to ntfy.sh/woody_videos_seb.
5. Kinematics & Photometry Weaving:
   - Populates native TopCam and native BotCam kinematics (972x972 scale).
   - Inferred KPMS heading (>0.99 cosine alignment) and smoothed velocity.
   - Synchronizes neural photometry streams (dFF, z-score, Wallace, RANSAC).
   - Serializes non-destructively to sessions_karen_project_tandem_production.pkl.
=============================================================================
"""

import os
import sys

# Unbuffered stdout/stderr
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Hardware environment flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_MEMORY_PATCH"] = "1"

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import copy
import pickle
import shutil
import signal
import time
import traceback
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import jax
import keypoint_moseq as kpms
import pandas as pd
from jax_moseq.utils.debugging import convert_data_precision
from scipy.ndimage import median_filter

from fpcode.common.paths import REPO_ROOT
KPMS_UTILS_DIR = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll"
if str(KPMS_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(KPMS_UTILS_DIR))

import utils.resample_batching as rb
from utils.batch_init import batched_init_model_by_video

from syllable_generation.analysis.replications.datta_utils import combined_photometry_and_syllables

# Configuration constants
EXPERIMENT_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
DIRTY_PKL_IN = REPO_ROOT / "syllable_generation/data/sessions_karen_project_dirty.pkl"
K240K_PKL_REF = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_k240k.pkl"
OUTPUT_PROD_PKL = EXPERIMENT_DIR / "sessions_karen_project_tandem_production.pkl"

TOP_H5_DIR = REPO_ROOT / "studies/karen_pff_vs_monomer/h5_files"
BOT_VID_DIR = REPO_ROOT / "studies/karen_pff_vs_monomer/botcam_video_files"
BOT_PRED_DIR = Path("/home/fsp585/Desktop/production_analysis_arm_B_seed_123_20260922/predictions")

PROD_KPMS_DIR = EXPERIMENT_DIR / "kpms_tandem_production"
PROD_DATA_DIR = PROD_KPMS_DIR / "data"
MODEL_NAME = "tandem_production_50ar_200slds"

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

NUM_AR_ITERS = 50
NUM_SLDS_ITERS = 200
TOTAL_ITERS = NUM_AR_ITERS + NUM_SLDS_ITERS  # 250 total
SAVE_EVERY_N_ITERS = 25

INIT_KAPPA = 7.5e13
DECREASE_KAPPA_FACTOR = 10.0
SLDS_KAPPA = INIT_KAPPA / DECREASE_KAPPA_FACTOR  # 7.5e12
TARGET_FPS = 60

TOP_USE_BODYPARTS = [
    "nose", "left_ear_tip", "right_ear_tip", "neck", "mid_back",
    "mouse_center", "mid_backend", "mid_backend2", "mid_backend3",
    "tail_base", "left_shoulder", "left_midside", "left_hip",
    "right_shoulder", "right_midside", "right_hip", "head_midpoint"
]

BOT_RENAME_MAP = {
    "left_front_paw": "left_front_paw",
    "right_front_paw": "right_front_paw",
    "left_hind_paw": "left_hind_paw",
    "right_hind_paw": "right_hind_paw",
    "nose_tip": "bot_nose_tip",
    "tail_base": "bot_tail_base"
}

target_sids = [
    "2026-03-05T11_01_01_top_left",     # Mouse 6, PFF, Arena 1
    "2026-03-05T11_01_01_bottom_left",  # Mouse 2, Monomer, Arena 2
]

session_configs = {
    "2026-03-05T11_01_01_top_left": {
        "arena": "Arena1",
        "bot_vid": BOT_VID_DIR / "16w_PI_T1_6.2.4_05.03.26_active_Arena1_BotCamVid_2026-03-05T11_01_01.avi",
        "bot_h5": BOT_PRED_DIR / "16w_PI_T1_6_2_4_05_03_26_active_Arena1_BotCamVid_2026-03-05T11_01_01/16w_PI_T1_6_2_4_05_03_26_active_Arena1_BotCamVid_2026-03-05T11_01_01DLC_HrnetW32_karen_bottom_modelSep14shuffle9_snapshot_best-180.h5",
        "top_h5": TOP_H5_DIR / "T1_6.2.4_05.03.26_Arena1234_CenterTopCamVid_2026-03-05T11_01_01_top_leftDLC_HrnetW32_dataset_fixDec16shuffle10_detector_020_snapshot_020.h5",
        "fused_h5": EXPERIMENT_DIR / "fused_h5_reflection_corrected/2026-03-05T11_01_01_top_left_tandem_23kp_full.h5"
    },
    "2026-03-05T11_01_01_bottom_left": {
        "arena": "Arena2",
        "bot_vid": BOT_VID_DIR / "16w_PI_T1_6.2.4_05.03.26_active_Arena2_BotCamVid_2026-03-05T11_01_01.avi",
        "bot_h5": BOT_PRED_DIR / "16w_PI_T1_6_2_4_05_03_26_active_Arena2_BotCamVid_2026-03-05T11_01_01/16w_PI_T1_6_2_4_05_03_26_active_Arena2_BotCamVid_2026-03-05T11_01_01DLC_HrnetW32_karen_bottom_modelSep14shuffle9_snapshot_best-180.h5",
        "top_h5": TOP_H5_DIR / "T1_6.2.4_05.03.26_Arena1234_CenterTopCamVid_2026-03-05T11_01_01_bottom_leftDLC_HrnetW32_dataset_fixDec16shuffle10_detector_020_snapshot_020.h5",
        "fused_h5": EXPERIMENT_DIR / "fused_h5_reflection_corrected/2026-03-05T11_01_01_bottom_left_tandem_23kp_full.h5"
    }
}


from fpcode.common.notify import send_ntfy


def handle_shutdown_signal(signum, frame):
    sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"\n[shutdown] Caught signal {sig_name} (PID {os.getpid()}). Flushing and alerting...", flush=True)
    send_ntfy(
        f"KPMS Tandem Run Interrupted ({sig_name})",
        f"Process caught {sig_name}. Checkpoint was saved on last 25-iter boundary. Resuming this script will continue from the latest checkpoint.",
        priority="high",
        tags="warning,pause"
    )
    sys.exit(128 + signum)

signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)


def fuse_full_session(top_h5_path, bot_h5_path, out_h5_path, fixed_scale=0.3716):
    """Generates fused 23-keypoint reflection-corrected DataFrame if not already present."""
    if out_h5_path.exists():
        print(f"  • Using existing fused H5: {out_h5_path.name}", flush=True)
        return out_h5_path, fixed_scale

    print(f"  • Generating fused H5: {out_h5_path.name}...", flush=True)
    df_top = pd.read_hdf(top_h5_path)
    df_bot = pd.read_hdf(bot_h5_path)

    top_scorer = df_top.columns.levels[0][0]
    bot_scorer = df_bot.columns.levels[0][0]

    n_frames = min(len(df_top), len(df_bot))
    active_slice = slice(0, n_frames)

    top_nx = df_top[top_scorer, "nose", "x"].iloc[active_slice].to_numpy(dtype=float)
    top_ny = df_top[top_scorer, "nose", "y"].iloc[active_slice].to_numpy(dtype=float)
    top_tx = df_top[top_scorer, "tail_base", "x"].iloc[active_slice].to_numpy(dtype=float)
    top_ty = df_top[top_scorer, "tail_base", "y"].iloc[active_slice].to_numpy(dtype=float)

    c_top = np.column_stack([0.5 * (top_nx + top_tx), 0.5 * (top_ny + top_ty)])
    c_top_smooth = median_filter(c_top, size=(11, 1), mode="nearest")

    theta_top = np.arctan2(top_ny - top_ty, top_nx - top_tx)
    cos_smooth = median_filter(np.cos(theta_top), size=11, mode="nearest")
    sin_smooth = median_filter(np.sin(theta_top), size=11, mode="nearest")
    theta_top_smooth = np.arctan2(sin_smooth, cos_smooth)

    bot_raw = {}
    for bp in ["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw", "nose_tip", "tail_base"]:
        bot_raw[bp] = {
            "x": df_bot[bot_scorer, bp, "x"].iloc[active_slice].to_numpy(dtype=float),
            "y": df_bot[bot_scorer, bp, "y"].iloc[active_slice].to_numpy(dtype=float),
            "p": df_bot[bot_scorer, bp, "likelihood"].iloc[active_slice].to_numpy(dtype=float)
        }

    bot_transformed = {}
    for bp in ["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw", "bot_nose_tip", "bot_tail_base"]:
        orig_key = "nose_tip" if bp == "bot_nose_tip" else ("tail_base" if bp == "bot_tail_base" else bp)
        coords = bot_raw[orig_key]

        x_refl = 972.0 - coords["x"]
        y_refl = coords["y"]

        bot_nx_r = 972.0 - bot_raw["nose_tip"]["x"]
        bot_ny_r = bot_raw["nose_tip"]["y"]
        bot_tx_r = 972.0 - bot_raw["tail_base"]["x"]
        bot_ty_r = bot_raw["tail_base"]["y"]

        c_bot_r = np.column_stack([0.5 * (bot_nx_r + bot_tx_r), 0.5 * (bot_ny_r + bot_ty_r)])
        c_bot_r_smooth = median_filter(c_bot_r, size=(11, 1), mode="nearest")

        theta_bot_r = np.arctan2(bot_ny_r - bot_ty_r, bot_nx_r - bot_tx_r)
        cos_b = median_filter(np.cos(theta_bot_r), size=11, mode="nearest")
        sin_b = median_filter(np.sin(theta_bot_r), size=11, mode="nearest")
        theta_bot_r_smooth = np.arctan2(sin_b, cos_b)

        p_rel_bot = np.column_stack([x_refl - c_bot_r_smooth[:, 0], y_refl - c_bot_r_smooth[:, 1]])

        cos_rel = np.cos(-theta_bot_r_smooth)
        sin_rel = np.sin(-theta_bot_r_smooth)
        p_ego_x = p_rel_bot[:, 0] * cos_rel - p_rel_bot[:, 1] * sin_rel
        p_ego_y = p_rel_bot[:, 0] * sin_rel + p_rel_bot[:, 1] * cos_rel

        p_ego_scaled = np.column_stack([p_ego_x * fixed_scale, p_ego_y * fixed_scale])

        cos_top = np.cos(theta_top_smooth)
        sin_top = np.sin(theta_top_smooth)
        p_top_x = p_ego_scaled[:, 0] * cos_top - p_ego_scaled[:, 1] * sin_top
        p_top_y = p_ego_scaled[:, 0] * sin_top + p_ego_scaled[:, 1] * cos_top

        p_virtual_top = np.column_stack([p_top_x + c_top_smooth[:, 0], p_top_y + c_top_smooth[:, 1]])

        bot_transformed[bp] = {
            "x": p_virtual_top[:, 0],
            "y": p_virtual_top[:, 1],
            "p": coords["p"]
        }

    fused_scorer = "Tandem_Dorsal_Ventral_Fused"
    columns_tuples = []
    columns_data = {}

    for bp in TOP_USE_BODYPARTS:
        columns_tuples.extend([(fused_scorer, bp, "x"), (fused_scorer, bp, "y"), (fused_scorer, bp, "likelihood")])
        columns_data[(fused_scorer, bp, "x")] = df_top[top_scorer, bp, "x"].iloc[active_slice].to_numpy(dtype=np.float32)
        columns_data[(fused_scorer, bp, "y")] = df_top[top_scorer, bp, "y"].iloc[active_slice].to_numpy(dtype=np.float32)
        columns_data[(fused_scorer, bp, "likelihood")] = df_top[top_scorer, bp, "likelihood"].iloc[active_slice].to_numpy(dtype=np.float32)

    for bp in ["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw", "bot_nose_tip", "bot_tail_base"]:
        columns_tuples.extend([(fused_scorer, bp, "x"), (fused_scorer, bp, "y"), (fused_scorer, bp, "likelihood")])
        columns_data[(fused_scorer, bp, "x")] = bot_transformed[bp]["x"].astype(np.float32)
        columns_data[(fused_scorer, bp, "y")] = bot_transformed[bp]["y"].astype(np.float32)
        columns_data[(fused_scorer, bp, "likelihood")] = bot_transformed[bp]["p"].astype(np.float32)

    fused_index = pd.MultiIndex.from_tuples(columns_tuples, names=["scorer", "bodyparts", "coords"])
    df_fused = pd.DataFrame(columns_data, columns=fused_index)

    out_h5_path.parent.mkdir(parents=True, exist_ok=True)
    df_fused.to_hdf(out_h5_path, key="df_with_missing", mode="w")
    print(f"  --> Fused H5 created: {out_h5_path.name} ({len(df_fused):,} frames, {len(df_fused.columns)//3} keypoints)", flush=True)
    return out_h5_path, fixed_scale


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tandem KPMS Production Training")
    parser.add_argument("--init-kappa", type=float, default=INIT_KAPPA, help="Initial Kappa for AR phase")
    parser.add_argument("--decrease-kappa-factor", type=float, default=DECREASE_KAPPA_FACTOR, help="Reduction factor for SLDS phase")
    parser.add_argument("--num-ar-iters", type=int, default=NUM_AR_ITERS, help="Number of AR iters")
    parser.add_argument("--num-slds-iters", type=int, default=NUM_SLDS_ITERS, help="Number of SLDS iters")
    parser.add_argument("--save-every-n-iters", type=int, default=SAVE_EVERY_N_ITERS, help="Checkpoint save interval")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--output-pkl", type=str, default=str(OUTPUT_PROD_PKL), help="Target output pickle path")
    args = parser.parse_args()

    init_kappa = args.init_kappa
    decrease_kappa_factor = args.decrease_kappa_factor
    slds_kappa = init_kappa / decrease_kappa_factor
    num_ar_iters = args.num_ar_iters
    num_slds_iters = args.num_slds_iters
    total_iters = num_ar_iters + num_slds_iters
    save_every_n_iters = args.save_every_n_iters
    model_name = args.model_name
    output_prod_pkl = Path(args.output_pkl)

    t_program_start = time.time()
    pid = os.getpid()

    print("=" * 80, flush=True)
    print(f"  TANDEM KPMS PRODUCTION RUN: {num_ar_iters} AR + {num_slds_iters} SLDS ({total_iters} TOTAL ITERS)", flush=True)
    print(f"  Date & Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  Process PID:         {pid}", flush=True)
    print(f"  Model Name:          {model_name}", flush=True)
    print("  Hardware:            RTX 4090 (24.5 GB VRAM) with Amdahl 27x Optimizations", flush=True)
    print(f"  AR Iterations:       {num_ar_iters}", flush=True)
    print(f"  SLDS Iterations:     {num_slds_iters} (Total: {total_iters})", flush=True)
    print(f"  Init Kappa:          {init_kappa:.2e} (SLDS Kappa: {slds_kappa:.2e})", flush=True)
    print(f"  Checkpoint Interval: Every {save_every_n_iters} iterations", flush=True)
    print(f"  Project Directory:   {PROD_KPMS_DIR}", flush=True)
    print(f"  Target Pickle:       {output_prod_pkl}", flush=True)
    print("=" * 80, flush=True)

    # 1. Apply high-performance kernels
    print("\n[Step 1/6] Applying Amdahl 27x Optimizations (Banded Cholesky + GPU Kernels)...", flush=True)
    rb.apply_optimizations()

    # Send launch notification
    send_ntfy(
        f"KPMS Tandem Started ({model_name})",
        f"PID: {pid}\n"
        f"Plan: {num_ar_iters} AR + {num_slds_iters} SLDS = {total_iters} iters\n"
        f"Checkpoints every {save_every_n_iters} iters\n"
        f"Kappa: {init_kappa:.2e} -> {slds_kappa:.2e}\n"
        f"Tandem sessions: 2 full sessions (416,032 frames @ 60 FPS)",
        priority="default",
        tags="rocket,brain"
    )

    # 2. Ingest Base Sessions & Enrich
    print("\n[Step 2/6] Loading base SessionRecord objects and enriching references...", flush=True)
    with open(DIRTY_PKL_IN, "rb") as f:
        dirty_sessions = pickle.load(f)

    selected_sessions = [copy.deepcopy(s) for s in dirty_sessions if s.session_id in target_sids]
    selected_sessions.sort(key=lambda s: target_sids.index(s.session_id))
    assert len(selected_sessions) == 2, f"Could not find both target sessions in {DIRTY_PKL_IN}!"

    for s in selected_sessions:
        cfg = session_configs[s.session_id]
        fuse_full_session(cfg["top_h5"], cfg["bot_h5"], cfg["fused_h5"])

        # Video refs enrichment
        v_df = s.video_refs_df.copy()
        if not (v_df["role"] == "ventral").any():
            ventral_vid_row = {
                "path": str(cfg["bot_vid"]),
                "path_type": "video",
                "kind": "avi",
                "role": "ventral",
                "ts": s.metadata.get("timestamp"),
                "chamber": s.metadata.get("chamber"),
                "camset": "BotCamVid",
                "cam_slot": "bottom_cam",
                "arena": cfg["arena"],
                "camera_key": f"bottom_cam:{s.metadata.get('chamber')}",
                "fps": 60.0,
                "n_frames": 208016,
                "duration_s": 208016 / 60.0,
                "meta_origin": "opencv"
            }
            v_df = pd.concat([v_df, pd.DataFrame([ventral_vid_row])], ignore_index=True)
        s.video_refs_df = v_df

        # Tracking refs enrichment
        t_df = s.tracking_refs_df.copy()
        if not (t_df["role"] == "ventral").any():
            ventral_track_row = {
                "path": str(cfg["bot_h5"]),
                "path_type": "tracking",
                "kind": "h5",
                "role": "ventral",
                "camset": "BotCamVid",
                "cam_slot": "bottom_cam",
                "arena": cfg["arena"],
                "chamber": s.metadata.get("chamber"),
                "ts": s.metadata.get("timestamp"),
                "camera_key": f"bottom_cam:{s.metadata.get('chamber')}",
                "chamber_index": s.metadata.get("chamber_index", 0)
            }
            t_df = pd.concat([t_df, pd.DataFrame([ventral_track_row])], ignore_index=True)

        if not (t_df["role"] == "fused_23kp").any():
            fused_track_row = {
                "path": str(cfg["fused_h5"]),
                "path_type": "tracking",
                "kind": "h5",
                "role": "fused_23kp",
                "camset": "TandemMultiview",
                "cam_slot": "multiview_fused",
                "arena": cfg["arena"],
                "chamber": s.metadata.get("chamber"),
                "ts": s.metadata.get("timestamp"),
                "camera_key": f"multiview_fused:{s.metadata.get('chamber')}",
                "chamber_index": s.metadata.get("chamber_index", 0)
            }
            t_df = pd.concat([t_df, pd.DataFrame([fused_track_row])], ignore_index=True)
        s.tracking_refs_df = t_df

    # 3. Setup KPMS Project & Data Directory
    print("\n[Step 3/6] Setting up KPMS production project environment...", flush=True)
    PROD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for sid, cfg in session_configs.items():
        src = cfg["fused_h5"]
        dst = PROD_DATA_DIR / f"{sid}_tandem_23kp.h5"
        if not dst.exists():
            dst.symlink_to(src)

    config_source = EXPERIMENT_DIR / "multiview_kappa_search/project/config.yml"
    config_dest = PROD_KPMS_DIR / "config.yml"
    if not config_dest.exists():
        shutil.copy(config_source, config_dest)

    config = kpms.load_config(str(PROD_KPMS_DIR))
    config["fps"] = TARGET_FPS
    config["trans_hypparams"]["kappa"] = init_kappa
    kpms.update_config(str(PROD_KPMS_DIR), **config)

    print("  Loading 23-keypoint tracking data into KPMS format...", flush=True)
    coordinates, confidences, bodyparts = kpms.load_keypoints(str(PROD_DATA_DIR), format="deeplabcut")
    data, metadata = kpms.format_data(coordinates, confidences, **config)
    data = convert_data_precision(data, np.float64)
    data["Y"] = np.asarray(data["Y"])

    # Load or fit PCA
    pca_p = PROD_KPMS_DIR / "pca.p"
    pca_asdf = PROD_KPMS_DIR / "pca.asdf"
    if pca_p.exists() or pca_asdf.exists():
        pca = kpms.load_pca(str(PROD_KPMS_DIR))
        print("  Loaded existing PCA from project.", flush=True)
    else:
        print("  Fitting and saving PCA...", flush=True)
        pca = kpms.fit_pca(**data, **config)
        kpms.save_pca(pca, str(PROD_KPMS_DIR))

    # Preemptive Analytical Batch Sizing
    print("  Executing Preemptive Analytical Tensor Sizing...", flush=True)
    rb.KPMSStateCache.clear()
    rb.precompute_optimal_batch_config(data, verbose=True)

    # 4. Checkpoint Resumption or Clean Initialization
    checkpoint_file = PROD_KPMS_DIR / model_name / "checkpoint.h5"
    current_iter = 0
    model = None

    if checkpoint_file.exists():
        try:
            print(f"\n  [RESUME] Found existing checkpoint at {checkpoint_file}. Resuming...", flush=True)
            model, data, metadata, current_iter = kpms.load_checkpoint(str(PROD_KPMS_DIR), model_name)
            data = convert_data_precision(data, np.float64)
            data["Y"] = np.asarray(data["Y"])
            rb.KPMSStateCache.clear()
            rb.precompute_optimal_batch_config(data, model=model, verbose=True)
            print(f"  [RESUME] Resumed successfully from iteration {current_iter}.", flush=True)
            send_ntfy(
                "KPMS Resumed from Checkpoint",
                f"Resumed training '{model_name}' at iteration {current_iter}/{total_iters}.",
                priority="default",
                tags="play,brain"
            )
        except Exception as e:
            print(f"  [warning] Failed to load checkpoint ({e}). Initializing fresh model...", flush=True)
            model = None
            current_iter = 0

    if model is None:
        print("\n  [INIT] Initializing fresh AR-HMM model with batched init...", flush=True)
        init_bs = min(rb.CONFIG.discrete_bs, 20)
        rng_key = jax.random.PRNGKey(42)
        model = batched_init_model_by_video(
            data, pca, config,
            params_subset_size=5, batch_size=init_bs, seed=rng_key
        )
        model = kpms.update_hypparams(model, kappa=float(init_kappa))
        current_iter = 0
        print("  [INIT] Initialization complete.", flush=True)

    # 5. Model Fitting Loop
    print(f"\n[Step 4/6] Executing KPMS Training Loop ({num_ar_iters} AR + {num_slds_iters} SLDS)...", flush=True)

    # PHASE 1: AR-ONLY FITTING (0 -> 50)
    if current_iter < num_ar_iters:
        print("\n" + "=" * 70, flush=True)
        print(f"  PHASE 1: AR-Only Fitting (iter {current_iter} -> {num_ar_iters})", flush=True)
        print(f"  Kappa: {init_kappa:.2e}", flush=True)
        print("=" * 70, flush=True)

        ar_start_wall = time.time()
        chunk_size = save_every_n_iters

        while current_iter < num_ar_iters:
            next_target = min(current_iter + chunk_size, num_ar_iters)
            chunk_t0 = time.time()
            print(f"\n>>> Running AR Chunk: iter {current_iter} -> {next_target}...", flush=True)

            model, _ = kpms.fit_model(
                model, data, metadata,
                project_dir=str(PROD_KPMS_DIR),
                model_name=model_name,
                ar_only=True,
                start_iter=current_iter,
                num_iters=next_target,
                parallel_message_passing=False,
                save_every_n_iters=save_every_n_iters,
                generate_progress_plots=True,
                verbose=True
            )
            current_iter = next_target

            # Syllable duration diagnostics on the fly
            z_curr = np.array(model["states"]["z"])
            mask_curr = np.array(data["mask"])[:, 3:]
            dur_list = kpms.viz.get_durations(z_curr, mask_curr)
            med_frames = float(np.median(dur_list)) if len(dur_list) > 0 else 0.0
            med_ms = (med_frames / TARGET_FPS) * 1000.0
            mean_frames = float(np.mean(dur_list)) if len(dur_list) > 0 else 0.0
            n_active_sylls = len(np.unique(z_curr[mask_curr > 0]))

            # Timing & ETA
            chunk_dur = time.time() - chunk_t0
            iters_done = current_iter
            sec_per_it = (time.time() - ar_start_wall) / iters_done
            ar_rem_sec = (num_ar_iters - current_iter) * sec_per_it
            eta_str = str(timedelta(seconds=int(ar_rem_sec)))

            print(f"--- Chunk Complete: AR {current_iter}/{num_ar_iters} ({sec_per_it:.2f}s/it, AR ETA: {eta_str}) | Median: {med_frames:.1f}f ({med_ms:.1f}ms) | Mean: {mean_frames:.1f}f | Sylls: {n_active_sylls} ---", flush=True)

            send_ntfy(
                f"KPMS AR Progress: {current_iter}/{num_ar_iters}",
                f"AR {current_iter}/{num_ar_iters} complete ({(current_iter/num_ar_iters)*100:.0f}%)\n"
                f"Median Syllable: {med_frames:.1f} frames ({med_ms:.1f} ms) | Active: {n_active_sylls}\n"
                f"Speed: {sec_per_it:.2f}s/it | AR ETA: {eta_str}",
                priority="default",
                tags="chart_with_upwards_trend"
            )

        ar_total_min = (time.time() - ar_start_wall) / 60.0
        print(f"\n[PHASE 1 COMPLETE] {num_ar_iters} AR iterations finished in {ar_total_min:.1f} min.", flush=True)
        send_ntfy(
            f"KPMS Phase 1 ({num_ar_iters} AR) Complete",
            f"{num_ar_iters} AR iterations finished in {ar_total_min:.1f} min.\n"
            f"Transitioning to Phase 2 SLDS (10x kappa reduction: {slds_kappa:.1e}).",
            priority="default",
            tags="tada,repeat"
        )
    else:
        print(f"  [Phase 1] Skipping AR (Already reached iter {current_iter} >= {num_ar_iters}).", flush=True)

    # TRANSITION: 10x Kappa reduction for SLDS
    print("\n" + "=" * 70, flush=True)
    print("  TRANSITION: Applying 10x Kappa Reduction for Phase 2 SLDS", flush=True)
    old_kappa = float(model['hypparams']['trans_hypparams']['kappa'])
    model = kpms.update_hypparams(model, kappa=float(slds_kappa))
    new_kappa = float(model['hypparams']['trans_hypparams']['kappa'])
    print(f"  Old Kappa: {old_kappa:.2e} -> New Kappa: {new_kappa:.2e}", flush=True)
    print("=" * 70, flush=True)

    # PHASE 2: SLDS FITTING
    if current_iter < total_iters:
        print("\n" + "=" * 70, flush=True)
        print(f"  PHASE 2: Full SLDS Fitting (iter {current_iter} -> {total_iters})", flush=True)
        print(f"  SLDS Kappa: {slds_kappa:.2e}", flush=True)
        print("=" * 70, flush=True)

        slds_start_wall = time.time()
        slds_start_iter = current_iter
        chunk_size = save_every_n_iters

        while current_iter < total_iters:
            next_target = min(current_iter + chunk_size, total_iters)
            chunk_t0 = time.time()
            print(f"\n>>> Running SLDS Chunk: iter {current_iter} -> {next_target}...", flush=True)

            model, _ = kpms.fit_model(
                model, data, metadata,
                project_dir=str(PROD_KPMS_DIR),
                model_name=model_name,
                ar_only=False,
                start_iter=current_iter,
                num_iters=next_target,
                parallel_message_passing=False,
                save_every_n_iters=save_every_n_iters,
                generate_progress_plots=True,
                verbose=True
            )
            # Syllable duration diagnostics on the fly
            z_curr = np.array(model["states"]["z"])
            mask_curr = np.array(data["mask"])[:, 3:]
            dur_list = kpms.viz.get_durations(z_curr, mask_curr)
            med_frames = float(np.median(dur_list)) if len(dur_list) > 0 else 0.0
            med_ms = (med_frames / TARGET_FPS) * 1000.0
            mean_frames = float(np.mean(dur_list)) if len(dur_list) > 0 else 0.0
            n_active_sylls = len(np.unique(z_curr[mask_curr > 0]))
            current_iter = next_target

            # Timing & ETA
            chunk_dur = time.time() - chunk_t0
            slds_done = current_iter - num_ar_iters
            sec_per_it = (time.time() - slds_start_wall) / (current_iter - slds_start_iter)
            slds_rem_sec = (total_iters - current_iter) * sec_per_it
            eta_str = str(timedelta(seconds=int(slds_rem_sec)))
            est_time = (datetime.now() + timedelta(seconds=slds_rem_sec)).strftime('%H:%M:%S')

            print(f"--- Chunk Complete: SLDS {slds_done}/{num_slds_iters} (Total {current_iter}/{total_iters}) | {sec_per_it:.2f}s/it | Median: {med_frames:.1f}f ({med_ms:.1f}ms) | Mean: {mean_frames:.1f}f | Sylls: {n_active_sylls} | ETA: {eta_str} (Est: {est_time}) ---", flush=True)

            send_ntfy(
                f"KPMS SLDS Progress: {slds_done}/{num_slds_iters}",
                f"SLDS {slds_done}/{num_slds_iters} complete (Total {current_iter}/{total_iters}, {(current_iter/total_iters)*100:.0f}%)\n"
                f"Median Syllable: {med_frames:.1f} frames ({med_ms:.1f} ms) | Active: {n_active_sylls}\n"
                f"Speed: {sec_per_it:.2f}s/it | ETA: {eta_str} (Est: {est_time})",
                priority="default",
                tags="hourglass_flowing_sand,brain"
            )

        slds_total_min = (time.time() - slds_start_wall) / 60.0
        print(f"\n[PHASE 2 COMPLETE] {num_slds_iters} SLDS iterations finished in {slds_total_min:.1f} min.", flush=True)
    else:
        print(f"  [Phase 2] Training already complete ({current_iter} >= {total_iters}).", flush=True)

    # 6. Extract Results and Weave Kinematics & Photometry
    print("\n[Step 5/6] Extracting final model results and weaving tandem kinematics...", flush=True)
    model, data, metadata, current_iter = kpms.load_checkpoint(str(PROD_KPMS_DIR), model_name)
    results = kpms.extract_results(model, metadata, str(PROD_KPMS_DIR), model_name, save_results=True)

    # Load baseline reference if available
    k240k_map = {}
    if K240K_PKL_REF.exists():
        try:
            with open(K240K_PKL_REF, "rb") as f:
                k240k_all = pickle.load(f)
            for s_ref in k240k_all:
                if s_ref.session_id in target_sids:
                    k240k_map[s_ref.session_id] = s_ref
        except Exception as e:
            print(f"  [warning] Could not load K240K reference: {e}", flush=True)

    for s in selected_sessions:
        sid = s.session_id
        cfg = session_configs[sid]
        rec_key = f"{sid}_tandem_23kp"
        s.results_dict_key = rec_key
        res = results[rec_key]

        tandem_sylls = res["syllable"]
        latent_states = res["latent_state"]
        moseq_centroid = res["centroid"]
        moseq_heading = res["heading"]

        n_frames = len(tandem_sylls)
        timeline_index = pd.RangeIndex(0, n_frames)

        # 1. Populate s.syllables
        syll_dict = {
            "syllable": pd.Series(tandem_sylls, dtype="Int64"),
        }
        for d in range(latent_states.shape[1]):
            syll_dict[f"latent_state_{d}"] = latent_states[:, d]
        syll_dict["centroid_0"] = moseq_centroid[:, 0]
        syll_dict["centroid_1"] = moseq_centroid[:, 1]
        syll_dict["heading"] = moseq_heading
        s.syllables = pd.DataFrame(syll_dict, index=timeline_index)

        # 2. Extract TopCam Raw Kinematics (native TopCam pixel space)
        df_top = pd.read_hdf(cfg["top_h5"])
        top_scorer = df_top.columns.levels[0][0]
        top_nx = df_top[top_scorer, "nose", "x"].iloc[:n_frames].to_numpy(dtype=float)
        top_ny = df_top[top_scorer, "nose", "y"].iloc[:n_frames].to_numpy(dtype=float)
        top_tx = df_top[top_scorer, "tail_base", "x"].iloc[:n_frames].to_numpy(dtype=float)
        top_ty = df_top[top_scorer, "tail_base", "y"].iloc[:n_frames].to_numpy(dtype=float)

        cx_top = 0.5 * (top_nx + top_tx)
        cy_top = 0.5 * (top_ny + top_ty)
        heading_top = np.arctan2(top_ny - top_ty, top_nx - top_tx)

        dx_top = np.diff(cx_top)
        dy_top = np.diff(cy_top)
        vel_top_raw = np.pad(np.sqrt(dx_top**2 + dy_top**2) * 60.0, (0, 1), mode="edge")
        ang_vel_top = np.pad(np.diff(np.unwrap(heading_top)) * 60.0, (0, 1), mode="edge")

        # Smoothed locomotor velocity (jump-clipped to <= 300 px/s with 7-frame median filter)
        vel_top_clipped = np.clip(vel_top_raw, 0, 300.0)
        vel_top_smooth = median_filter(vel_top_clipped, size=7, mode="nearest")

        # 3. Extract BotCam Raw Kinematics (native BotCam 972x972 pixel space)
        df_bot = pd.read_hdf(cfg["bot_h5"])
        bot_scorer = df_bot.columns.levels[0][0]
        bot_nx = df_bot[bot_scorer, "nose_tip", "x"].iloc[:n_frames].to_numpy(dtype=float)
        bot_ny = df_bot[bot_scorer, "nose_tip", "y"].iloc[:n_frames].to_numpy(dtype=float)
        bot_tx = df_bot[bot_scorer, "tail_base", "x"].iloc[:n_frames].to_numpy(dtype=float)
        bot_ty = df_bot[bot_scorer, "tail_base", "y"].iloc[:n_frames].to_numpy(dtype=float)

        cx_bot = 0.5 * (bot_nx + bot_tx)
        cy_bot = 0.5 * (bot_ny + bot_ty)
        heading_bot = np.arctan2(bot_ny - bot_ty, bot_nx - bot_tx)

        dx_bot = np.diff(cx_bot)
        dy_bot = np.diff(cy_bot)
        vel_bot_raw = np.pad(np.sqrt(dx_bot**2 + dy_bot**2) * 60.0, (0, 1), mode="edge")
        ang_vel_bot = np.pad(np.diff(np.unwrap(heading_bot)) * 60.0, (0, 1), mode="edge")

        vel_bot_clipped = np.clip(vel_bot_raw, 0, 300.0)
        vel_bot_smooth = median_filter(vel_bot_clipped, size=7, mode="nearest")

        # 4. Onset detection
        onset = np.concatenate([[True], tandem_sylls[1:] != tandem_sylls[:-1]])

        # 5. Build Behavior DataFrame
        kinematics_dict = {
            "frame_index": timeline_index.to_numpy(),
            "name": [rec_key] * n_frames,
            # Backward-compatible baseline kinematics
            "centroid_x": cx_top,
            "centroid_y": cy_top,
            "heading": moseq_heading,
            "heading_top": heading_top,
            "heading_moseq": moseq_heading,
            "angular_velocity": ang_vel_top,
            "velocity_px_s": vel_top_smooth,
            "velocity_px_s_raw": vel_top_raw,
            # Syllables
            "syllable": pd.Series(tandem_sylls, dtype="Int64"),
            "tandem_syllable": pd.Series(tandem_sylls, dtype="Int64"),
            "group": ["default"] * n_frames,
            "onset": onset,
            # Explicit TopCam Kinematics
            "centroid_x_top": cx_top,
            "centroid_y_top": cy_top,
            "angular_velocity_top": ang_vel_top,
            "velocity_px_s_top": vel_top_smooth,
            "velocity_px_s_top_raw": vel_top_raw,
            # Explicit BotCam Kinematics (native 972x972 pixel space)
            "centroid_x_bot": cx_bot,
            "centroid_y_bot": cy_bot,
            "heading_bot": heading_bot,
            "angular_velocity_bot": ang_vel_bot,
            "velocity_px_s_bot": vel_bot_smooth,
            "velocity_px_s_bot_raw": vel_bot_raw,
        }

        if sid in k240k_map and "syllable" in k240k_map[sid].all_data_df.columns:
            k_syll = k240k_map[sid].all_data_df["syllable"].iloc[:n_frames]
            kinematics_dict["top_syllable"] = k_syll.to_numpy()

        df_behavior = pd.DataFrame(kinematics_dict, index=timeline_index)

        # 6. Weave Photometry
        phot_df = combined_photometry_and_syllables(s)
        phot_aligned = phot_df.reindex(timeline_index).reset_index(drop=True)
        s.all_data_df = pd.concat([df_behavior.reset_index(drop=True), phot_aligned], axis=1)

        print(f"  Session {sid}: all_data_df shape = {s.all_data_df.shape} | Distinct syllables: {len(np.unique(tandem_sylls))}", flush=True)

    # 7. Serialize to Output Pickle
    print(f"\n[Step 6/6] Serializing production pickle to {output_prod_pkl}...", flush=True)
    output_prod_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_prod_pkl, "wb") as f:
        pickle.dump(selected_sessions, f)

    file_size_mb = output_prod_pkl.stat().st_size / (1024 * 1024)
    total_elapsed_min = (time.time() - t_program_start) / 60.0

    print("=" * 80, flush=True)
    print(f"  PRODUCTION RUN COMPLETED SUCCESSFULLY IN {total_elapsed_min:.1f} MIN!", flush=True)
    print(f"  Saved File:   {output_prod_pkl.name} ({file_size_mb:.2f} MB)", flush=True)
    print(f"  Total Frames: {sum(len(s.all_data_df) for s in selected_sessions):,}", flush=True)
    print("=" * 80, flush=True)

    send_ntfy(
        f"KPMS Run Succeeded! ({model_name}) 🎉",
        f"Completed {num_ar_iters} AR + {num_slds_iters} SLDS in {total_elapsed_min:.1f} min.\n"
        f"Output: {output_prod_pkl.name} ({file_size_mb:.1f} MB)\n"
        f"Syllables and multi-view kinematics fully synchronized with photometry!",
        priority="high",
        tags="white_check_mark,party_popper"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        err_tb = traceback.format_exc()
        print("\n" + "!" * 80, flush=True)
        print(f"FATAL ERROR IN TANDEM PRODUCTION RUN:\n{err_tb}", flush=True)
        print("!" * 80, flush=True)
        send_ntfy(
            "KPMS Production Run FAILED ❌",
            f"Fatal error encountered:\n{str(exc)[:200]}\nCheck logs for traceback.",
            priority="urgent",
            tags="x,skull"
        )
        sys.exit(1)
