#!/usr/bin/env python3
"""
fuse_multiview_cohort.py
------------------------
Fuses TopCam dorsal tracking (17 keypoints) with BotCam ventral tracking (6 keypoints)
via canonical egocentric body-frame transformation.
Produces 23-keypoint H5 files for the balanced 8-session cohort.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from fpcode.tracking.geometry import rot2d as _rotation_matrices

# Cohort specification: 4 Monomer + 4 PFF across 16w and 24w
TARGET_SESSIONS = [
    # 16w Monomer
    ("2026-03-05T11_01_01", "Arena2", "bottom_left", "16w_PI", "Monomer", "2"),
    ("2026-03-05T12_46_18", "Arena2", "bottom_left", "16w_PI", "Monomer", "8"),
    # 16w PFF
    ("2026-03-05T11_01_01", "Arena1", "top_left", "16w_PI", "PFF", "6"),
    ("2026-03-05T12_46_18", "Arena1", "top_left", "16w_PI", "PFF", "5"),
    # 24w Monomer
    ("2026-04-28T14_58_25", "Arena2", "bottom_left", "24w_PI", "Monomer", "17"),
    ("2026-04-28T13_21_05", "Arena1", "top_left", "24w_PI", "Monomer", "15"),
    # 24w PFF
    ("2026-04-28T14_58_25", "Arena1", "top_left", "24w_PI", "PFF", "19"),
    ("2026-04-28T13_21_05", "Arena2", "bottom_left", "24w_PI", "PFF", "16"),
]

TOP_H5_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/studies/karen_pff_vs_monomer/h5_files")
DLC_PROD_DIR = Path("/home/fsp585/Desktop/production_analysis_arm_B_seed_123_20260922/predictions")
OUT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/multiview_kappa_search/fused_h5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "tail_base": "bot_tail_base",
}

def rot2d(angle: np.ndarray) -> np.ndarray:
    return _rotation_matrices(angle, dtype=np.float32)

def fuse_session(ts: str, arena: str, chamber: str, condition: str, genotype: str, animal_id: str):
    print(f"\nProcessing {ts}_{chamber} ({condition}, {genotype}, Animal {animal_id})...")

    # 1. Locate TopCam H5
    top_candidates = list(TOP_H5_DIR.glob(f"*{ts}*{chamber}*.h5"))
    if not top_candidates:
        top_candidates = list(TOP_H5_DIR.glob(f"*{ts}*CenterTopCamVid*.h5"))
    if not top_candidates:
        raise FileNotFoundError(f"TopCam H5 not found for {ts} {chamber}")
    top_h5_path = top_candidates[0]

    # 2. Locate BotCam H5
    bot_candidates = list(DLC_PROD_DIR.glob(f"*{ts}*_{arena}_BotCamVid*/*snapshot_best-180.h5"))
    if not bot_candidates:
        # Check with dots instead of underscores or vice versa
        ts_clean = ts.replace(":", "_").replace("-", "_")
        bot_candidates = list(DLC_PROD_DIR.glob(f"*{arena}*/*snapshot_best-180.h5"))
        bot_candidates = [p for p in bot_candidates if ts in p.name or ts.replace(":", "_") in p.name or ts[:10].replace("-", ".") in p.name]
    if not bot_candidates:
        raise FileNotFoundError(f"BotCam H5 not found for {ts} {arena}")
    bot_h5_path = bot_candidates[0]

    print(f"  TopCam: {top_h5_path.name}")
    print(f"  BotCam: {bot_h5_path.parent.name}/{bot_h5_path.name}")

    df_top = pd.read_hdf(top_h5_path)
    df_bot = pd.read_hdf(bot_h5_path)

    top_scorer = df_top.columns.levels[0][0]
    bot_scorer = df_bot.columns.levels[0][0]

    # Slice active behavior frames: frame 25,000 to 180,000 (155,000 frames)
    # No downsampling (step = 1) -> 155,000 frames @ 60 FPS
    start_frame = 25000
    end_frame = min(180000, len(df_top), len(df_bot))
    step = 1

    active_slice = slice(start_frame, end_frame, step)
    n_frames = len(range(start_frame, end_frame, step))
    print(f"  Active slice [{start_frame}:{end_frame}:{step}] -> {n_frames} frames @ 60 FPS")

    # Extract TopCam dorsal anchors
    top_nose_x = df_top[top_scorer, "nose", "x"].iloc[active_slice].to_numpy(dtype=np.float32)
    top_nose_y = df_top[top_scorer, "nose", "y"].iloc[active_slice].to_numpy(dtype=np.float32)
    top_tail_x = df_top[top_scorer, "tail_base", "x"].iloc[active_slice].to_numpy(dtype=np.float32)
    top_tail_y = df_top[top_scorer, "tail_base", "y"].iloc[active_slice].to_numpy(dtype=np.float32)

    top_c_x = 0.5 * (top_nose_x + top_tail_x)
    top_c_y = 0.5 * (top_nose_y + top_tail_y)
    top_dx = top_nose_x - top_tail_x
    top_dy = top_nose_y - top_tail_y
    top_len = np.sqrt(top_dx**2 + top_dy**2)
    theta_top = np.arctan2(top_dy, top_dx)

    # Extract BotCam ventral anchors
    bot_nose_x = df_bot[bot_scorer, "nose_tip", "x"].iloc[active_slice].to_numpy(dtype=np.float32)
    bot_nose_y = df_bot[bot_scorer, "nose_tip", "y"].iloc[active_slice].to_numpy(dtype=np.float32)
    bot_tail_x = df_bot[bot_scorer, "tail_base", "x"].iloc[active_slice].to_numpy(dtype=np.float32)
    bot_tail_y = df_bot[bot_scorer, "tail_base", "y"].iloc[active_slice].to_numpy(dtype=np.float32)

    bot_c_x = 0.5 * (bot_nose_x + bot_tail_x)
    bot_c_y = 0.5 * (bot_nose_y + bot_tail_y)
    bot_dx = bot_nose_x - bot_tail_x
    bot_dy = bot_nose_y - bot_tail_y
    bot_len = np.sqrt(bot_dx**2 + bot_dy**2)
    theta_bot = np.arctan2(bot_dy, bot_dx)

    # Robust scale factor (avoid division by 0)
    valid_len = (top_len > 10.0) & (bot_len > 10.0)
    median_scale = np.median(top_len[valid_len] / bot_len[valid_len])
    print(f"  Median dorsal/ventral scale factor: {median_scale:.4f}")

    # Rotation matrices: R_bot_inv (N, 2, 2) and R_top (N, 2, 2)
    R_bot_inv = rot2d(-theta_bot)
    R_top     = rot2d(theta_top)

    # Build MultiIndex dictionary for the fused DataFrame
    fused_dict = {}
    fused_scorer = "fused_multiview"

    # 1. Add TopCam 17 bodyparts
    for bp in TOP_USE_BODYPARTS:
        fused_dict[(fused_scorer, bp, "x")] = df_top[top_scorer, bp, "x"].iloc[active_slice].to_numpy(dtype=np.float32)
        fused_dict[(fused_scorer, bp, "y")] = df_top[top_scorer, bp, "y"].iloc[active_slice].to_numpy(dtype=np.float32)
        fused_dict[(fused_scorer, bp, "likelihood")] = df_top[top_scorer, bp, "likelihood"].iloc[active_slice].to_numpy(dtype=np.float32)

    # 2. Add transformed BotCam 6 bodyparts
    for bot_bp, new_bp in BOT_RENAME_MAP.items():
        bx = df_bot[bot_scorer, bot_bp, "x"].iloc[active_slice].to_numpy(dtype=np.float32)
        by = df_bot[bot_scorer, bot_bp, "y"].iloc[active_slice].to_numpy(dtype=np.float32)
        blike = df_bot[bot_scorer, bot_bp, "likelihood"].iloc[active_slice].to_numpy(dtype=np.float32)

        # Centering relative to ventral mid
        dx = bx - bot_c_x
        dy = by - bot_c_y
        vec_bot = np.stack([dx, dy], axis=-1)  # (N, 2)

        # Egocentric rotate: R_bot_inv @ vec_bot
        ego_x = R_bot_inv[:, 0, 0] * vec_bot[:, 0] + R_bot_inv[:, 0, 1] * vec_bot[:, 1]
        ego_y = R_bot_inv[:, 1, 0] * vec_bot[:, 0] + R_bot_inv[:, 1, 1] * vec_bot[:, 1]

        # Scale
        ego_x *= median_scale
        ego_y *= median_scale

        # Top rotate & translocate
        top_proj_x = top_c_x + (R_top[:, 0, 0] * ego_x + R_top[:, 0, 1] * ego_y)
        top_proj_y = top_c_y + (R_top[:, 1, 0] * ego_x + R_top[:, 1, 1] * ego_y)

        fused_dict[(fused_scorer, new_bp, "x")] = top_proj_x
        fused_dict[(fused_scorer, new_bp, "y")] = top_proj_y
        fused_dict[(fused_scorer, new_bp, "likelihood")] = blike

    # Build DataFrame
    columns = pd.MultiIndex.from_tuples(list(fused_dict.keys()), names=["scorer", "bodyparts", "coords"])
    df_fused = pd.DataFrame(fused_dict, index=np.arange(n_frames))
    df_fused.columns = columns

    out_h5 = OUT_DIR / f"{ts}_{chamber}_multiview_23kp.h5"
    df_fused.to_hdf(out_h5, key="df_with_missing", mode="w", format="table")
    print(f"  Saved fused H5: {out_h5.name} (shape: {df_fused.shape})")
    return out_h5

def main():
    print("=== STARTING BATCH MULTIVIEW FUSION ===")
    results = []
    for item in TARGET_SESSIONS:
        h5_path = fuse_session(*item)
        results.append(h5_path)
    print(f"\nSuccessfully fused all {len(results)} sessions into {OUT_DIR}!")

if __name__ == "__main__":
    main()
