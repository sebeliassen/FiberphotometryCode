#!/usr/bin/env python3
"""
fuse_reflection_corrected_sessions.py
=============================================================================
Fuses TopCam dorsal tracking (17 keypoints) with BotCam ventral tracking (6 keypoints)
for representative test sessions using the 4 Golden Rules:

1. Fixed robust scale per session from reliable frames.
2. Pre-heading horizontal reflection correction (x -> 972 - x) on BotCam,
   with bilateral label swap (left_paw <-> right_paw).
3. Anatomical nose-tail midpoint and heading direction in both views.
4. Circular smoothing on reference headings & median smoothing on centroids,
   leaving individual paw coordinates 100% raw, un-smoothed, and unadulterated.

SAFETY:
- Strictly writes to: fused_h5_reflection_corrected/
- Zero side effects on existing projects, models, or pickles.
=============================================================================
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from fpcode.common.paths import get_repo_root
from fpcode.tracking.geometry import rot2d as _rotation_matrices

REPO_ROOT = Path(str(get_repo_root()))
TOP_H5_DIR = REPO_ROOT / 'data/raw/karen_pff_vs_monomer/h5_files'
BOT_PROD_DIR = Path("/home/fsp585/Desktop/production_analysis_arm_B_seed_123_20260922/predictions")

OUT_DIR = REPO_ROOT / 'studies/multiview_experiments/fused_h5_reflection_corrected'
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

def smooth_heading_circular(theta: np.ndarray, window: int = 9) -> np.ndarray:
    sin_s = median_filter(np.sin(theta), size=window, mode="nearest")
    cos_s = median_filter(np.cos(theta), size=window, mode="nearest")
    return np.arctan2(sin_s, cos_s)

def smooth_centroid_median(c: np.ndarray, window: int = 9) -> np.ndarray:
    return np.column_stack([
        median_filter(c[:, 0], size=window, mode="nearest"),
        median_filter(c[:, 1], size=window, mode="nearest")
    ])

def fuse_single_session(
    ts: str,
    arena: str,
    chamber: str,
    top_h5_path: Path,
    bot_h5_path: Path,
    active_range: tuple = (25000, 180000),
    arena_width: float = 972.0
) -> Path:
    print(f"\n--- Fusing Session: {ts} {arena} ({chamber}) ---")
    print(f"  TopCam: {top_h5_path.name}")
    print(f"  BotCam: {bot_h5_path.parent.name}/{bot_h5_path.name}")

    df_top = pd.read_hdf(top_h5_path)
    df_bot = pd.read_hdf(bot_h5_path)

    top_scorer = df_top.columns.levels[0][0]
    bot_scorer = df_bot.columns.levels[0][0]

    # Slice active behaviors
    start_f, end_f = active_range
    end_f = min(end_f, len(df_top), len(df_bot))
    active_slice = slice(start_f, end_f)
    n_frames = end_f - start_f
    print(f"  Active frames: [{start_f}:{end_f}] -> {n_frames:,} frames @ 60 FPS")

    # -------------------------------------------------------------------------
    # RULE 2: Correct Horizontal Camera Reflection & Bilateral Label Swap
    # -------------------------------------------------------------------------
    # Extract BotCam landmarks and horizontally mirror x-coordinates
    bot_raw_coords = {}
    for bp in df_bot[bot_scorer].columns.levels[0]:
        bx = df_bot[bot_scorer, bp, "x"].iloc[active_slice].to_numpy(dtype=np.float32)
        by = df_bot[bot_scorer, bp, "y"].iloc[active_slice].to_numpy(dtype=np.float32)
        blike = df_bot[bot_scorer, bp, "likelihood"].iloc[active_slice].to_numpy(dtype=np.float32)
        # Invert horizontal axis
        bx_corr = arena_width - bx
        bot_raw_coords[bp] = {"x": bx_corr, "y": by, "p": blike}

    # Swap left and right paws after horizontal flip
    bot_corr_coords = {}
    bot_corr_coords["left_front_paw"] = bot_raw_coords["right_front_paw"]
    bot_corr_coords["right_front_paw"] = bot_raw_coords["left_front_paw"]
    bot_corr_coords["left_hind_paw"] = bot_raw_coords["right_hind_paw"]
    bot_corr_coords["right_hind_paw"] = bot_raw_coords["left_hind_paw"]
    bot_corr_coords["nose_tip"] = bot_raw_coords["nose_tip"]
    bot_corr_coords["tail_base"] = bot_raw_coords["tail_base"]

    # -------------------------------------------------------------------------
    # Extract TopCam landmarks
    # -------------------------------------------------------------------------
    top_nose_x = df_top[top_scorer, "nose", "x"].iloc[active_slice].to_numpy(dtype=np.float32)
    top_nose_y = df_top[top_scorer, "nose", "y"].iloc[active_slice].to_numpy(dtype=np.float32)
    top_tail_x = df_top[top_scorer, "tail_base", "x"].iloc[active_slice].to_numpy(dtype=np.float32)
    top_tail_y = df_top[top_scorer, "tail_base", "y"].iloc[active_slice].to_numpy(dtype=np.float32)

    bot_nose_x = bot_corr_coords["nose_tip"]["x"]
    bot_nose_y = bot_corr_coords["nose_tip"]["y"]
    bot_tail_x = bot_corr_coords["tail_base"]["x"]
    bot_tail_y = bot_corr_coords["tail_base"]["y"]

    # -------------------------------------------------------------------------
    # RULE 1: Robust Fixed Session Scale
    # -------------------------------------------------------------------------
    len_top = np.sqrt((top_nose_x - top_tail_x)**2 + (top_nose_y - top_tail_y)**2)
    len_bot = np.sqrt((bot_nose_x - bot_tail_x)**2 + (bot_nose_y - bot_tail_y)**2)
    reliable = (len_top > 20.0) & (len_bot > 40.0)
    fixed_scale = float(np.nanmedian(len_top[reliable]) / np.nanmedian(len_bot[reliable]))
    print(f"  --> Fixed scale: s = {fixed_scale:.4f} (median Top={np.nanmedian(len_top[reliable]):.1f} px, Bot={np.nanmedian(len_bot[reliable]):.1f} px)")

    # -------------------------------------------------------------------------
    # RULE 3: Anatomical Nose-Tail Midpoint & Direction
    # -------------------------------------------------------------------------
    c_top_raw = np.column_stack([0.5 * (top_nose_x + top_tail_x), 0.5 * (top_nose_y + top_tail_y)])
    c_bot_raw = np.column_stack([0.5 * (bot_nose_x + bot_tail_x), 0.5 * (bot_nose_y + bot_tail_y)])

    theta_top_raw = np.arctan2(top_nose_y - top_tail_y, top_nose_x - top_tail_x)
    theta_bot_raw = np.arctan2(bot_nose_y - bot_tail_y, bot_nose_x - bot_tail_x)

    # -------------------------------------------------------------------------
    # RULE 4: Circular Smoothing on Heading & Median Smoothing on Centroid
    # -------------------------------------------------------------------------
    theta_top = smooth_heading_circular(theta_top_raw, window=9)
    theta_bot = smooth_heading_circular(theta_bot_raw, window=9)
    c_top = smooth_centroid_median(c_top_raw, window=9)
    c_bot = smooth_centroid_median(c_bot_raw, window=9)

    d_th = np.arctan2(np.sin(theta_top - theta_bot), np.cos(theta_top - theta_bot))
    print(f"  --> Mean absolute heading alignment: {np.degrees(np.nanmean(np.abs(d_th[reliable]))):.2f}°")

    # Rotation matrices
    R_bot_inv = rot2d(-theta_bot)
    R_top = rot2d(theta_top)

    # -------------------------------------------------------------------------
    # 5. Assemble MultiIndex Fused DataFrame
    # -------------------------------------------------------------------------
    fused_dict = {}
    fused_scorer = "fused_multiview"

    # Add TopCam 17 dorsal landmarks (unaltered)
    for bp in TOP_USE_BODYPARTS:
        fused_dict[(fused_scorer, bp, "x")] = df_top[top_scorer, bp, "x"].iloc[active_slice].to_numpy(dtype=np.float32)
        fused_dict[(fused_scorer, bp, "y")] = df_top[top_scorer, bp, "y"].iloc[active_slice].to_numpy(dtype=np.float32)
        fused_dict[(fused_scorer, bp, "likelihood")] = df_top[top_scorer, bp, "likelihood"].iloc[active_slice].to_numpy(dtype=np.float32)

    # Translocate BotCam 6 ventral landmarks (paws are raw and un-smoothed!)
    for bot_bp, new_bp in BOT_RENAME_MAP.items():
        bx = bot_corr_coords[bot_bp]["x"]
        by = bot_corr_coords[bot_bp]["y"]
        bp_like = bot_corr_coords[bot_bp]["p"]

        # Center relative to smoothed BotCam centroid
        dx = bx - c_bot[:, 0]
        dy = by - c_bot[:, 1]
        vec_bot = np.stack([dx, dy], axis=-1)

        # Rotate to egocentric frame: R_bot_inv @ vec_bot
        ego_x = R_bot_inv[:, 0, 0] * vec_bot[:, 0] + R_bot_inv[:, 0, 1] * vec_bot[:, 1]
        ego_y = R_bot_inv[:, 1, 0] * vec_bot[:, 0] + R_bot_inv[:, 1, 1] * vec_bot[:, 1]

        # Scale
        ego_x *= fixed_scale
        ego_y *= fixed_scale

        # Rotate to TopCam virtual arena frame
        proj_x = c_top[:, 0] + (R_top[:, 0, 0] * ego_x + R_top[:, 0, 1] * ego_y)
        proj_y = c_top[:, 1] + (R_top[:, 1, 0] * ego_x + R_top[:, 1, 1] * ego_y)

        fused_dict[(fused_scorer, new_bp, "x")] = proj_x
        fused_dict[(fused_scorer, new_bp, "y")] = proj_y
        fused_dict[(fused_scorer, new_bp, "likelihood")] = bp_like

    columns = pd.MultiIndex.from_tuples(list(fused_dict.keys()), names=["scorer", "bodyparts", "coords"])
    df_fused = pd.DataFrame(fused_dict, index=np.arange(n_frames))
    df_fused.columns = columns

    # Save to HDF5
    out_stem = f"{ts}_{chamber}_tandem_23kp_refl_corr.h5"
    out_path = OUT_DIR / out_stem
    df_fused.to_hdf(str(out_path), key="df_with_missing", mode="w", format="fixed")
    print(f"  [SAVED] {out_path.name} ({n_frames:,} frames, 23 keypoints, {out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path

def main():
    print("=" * 80)
    print("🚀 GENERATING REFLECTION-CORRECTED TANDEM COORDINATES (3 TEST SESSIONS)")
    print(f"Output Directory: {OUT_DIR}")
    print("=" * 80)

    test_sessions = [
        # Session 1: 16w_PI T1 Arena 1 (top_left, Mouse 6, PFF)
        ("2026-03-05T11_01_01", "Arena1", "top_left"),
        # Session 2: 16w_PI T1 Arena 2 (bottom_left, Mouse 2, Monomer)
        ("2026-03-05T11_01_01", "Arena2", "bottom_left"),
        # Session 3: 16w_PI T2 Arena 3 (bottom_right, Mouse 9, Monomer)
        ("2026-03-05T12_46_18", "Arena3", "bottom_right"),
    ]

    fused_paths = []
    for ts, arena, chamber in test_sessions:
        top_candidates = list(TOP_H5_DIR.glob(f"*{ts}*{chamber}*.h5"))
        bot_candidates = list(BOT_PROD_DIR.glob(f"*{arena}*/*snapshot_best-180.h5"))
        bot_candidates = [p for p in bot_candidates if ts in p.name or ts.replace(":", "_") in p.name or ts[:10].replace("-", ".") in p.name]

        if not top_candidates or not bot_candidates:
            print(f"ERROR: Files not found for {ts} {arena} {chamber}")
            continue

        p = fuse_single_session(
            ts=ts,
            arena=arena,
            chamber=chamber,
            top_h5_path=top_candidates[0],
            bot_h5_path=bot_candidates[0],
        )
        fused_paths.append(p)

    print("\n" + "=" * 80)
    print(f"🎉 GENERATED {len(fused_paths)} / {len(test_sessions)} FUSED SESSIONS SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
