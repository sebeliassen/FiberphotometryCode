#!/usr/bin/env python3
"""
02_fuse_dorsal_ventral.py
-------------------------
Implements the canonical egocentric body-centered fusion algorithm:
1. Loads TopCam dorsal predictions and BotCam ventral predictions for matched sessions.
2. Uses clock_sync.py to align frames based on hardware FP3002_System_Timestamp.
3. Extracts body axes:
   - TopCam: tail_base -> nose / head_midpoint
   - BotCam: tail_base -> nose_tip
4. Computes median rostral-caudal body scale ratio s = L_top / L_bot.
5. Normalizes BotCam ventral paws into canonical egocentric frame.
6. Projects egocentric paws into TopCam virtual arena coordinate space.
7. Exports unified 21-keypoint HDF5 files formatted for standard Keypoint-MoSeq.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.common.paths import get_repo_root

REPO_ROOT = Path(str(get_repo_root()))
EXPERIMENT_DIR = (get_repo_root() / 'studies/multiview_experiments')
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from clock_sync import align_tracking_dataframes

MANIFEST_FILE = EXPERIMENT_DIR / "paired_sessions_manifest.json"
OUTPUT_H5_DIR = EXPERIMENT_DIR / "fused_h5_predictions"
OUTPUT_H5_DIR.mkdir(parents=True, exist_ok=True)

TOPCAM_BPS = [
    "nose", "left_ear_tip", "right_ear_tip", "neck", "mid_back",
    "mouse_center", "mid_backend", "mid_backend2", "mid_backend3",
    "tail_base", "left_shoulder", "left_midside", "left_hip",
    "right_shoulder", "right_midside", "right_hip", "head_midpoint"
]

BOTCAM_PAWS = [
    "left_front_paw", "right_front_paw",
    "left_hind_paw", "right_hind_paw"
]

def compute_heading_and_centroid(df, scorer, nose_bp, tail_bp, center_bp=None):
    """
    Computes frame-by-frame 2D centroid and heading angle from nose and tail.
    """
    nose_x = df[scorer][nose_bp]['x'].values
    nose_y = df[scorer][nose_bp]['y'].values
    tail_x = df[scorer][tail_bp]['x'].values
    tail_y = df[scorer][tail_bp]['y'].values

    if center_bp and center_bp in df[scorer].columns.levels[0]:
        cx = df[scorer][center_bp]['x'].values
        cy = df[scorer][center_bp]['y'].values
    else:
        cx = 0.5 * (nose_x + tail_x)
        cy = 0.5 * (nose_y + tail_y)

    dx = nose_x - tail_x
    dy = nose_y - tail_y
    heading = np.arctan2(dy, dx)
    length = np.sqrt(dx**2 + dy**2)

    return np.stack([cx, cy], axis=-1), heading, length

def fuse_session_pair(
    top_h5_path: str | Path,
    bot_h5_path: str | Path,
    out_h5_path: str | Path,
    top_ts: np.ndarray | None = None,
    bot_ts: np.ndarray | None = None,
    max_frames: int | None = None
):
    """
    Fuses TopCam and BotCam HDF5 tracking files into a single unified 21-keypoint HDF5 file.
    Uses hardware clock synchronization when timestamps are provided.
    """
    df_top = pd.read_hdf(top_h5_path)
    df_bot = pd.read_hdf(bot_h5_path)

    scorer_top = df_top.columns.levels[0][0]
    scorer_bot = df_bot.columns.levels[0][0]

    # Time-lock BotCam to TopCam using clock synchronization
    df_top, df_bot, sync_stats = align_tracking_dataframes(
        df_top, df_bot, top_ts=top_ts, bot_ts=bot_ts
    )

    if max_frames is not None:
        df_top = df_top.iloc[:max_frames]
        df_bot = df_bot.iloc[:max_frames]

    # 1. Compute TopCam body frame
    c_top, theta_top, len_top = compute_heading_and_centroid(
        df_top, scorer_top, "nose", "tail_base", center_bp="mouse_center"
    )

    # 2. Compute BotCam body frame
    c_bot, theta_bot, len_bot = compute_heading_and_centroid(
        df_bot, scorer_bot, "nose_tip", "tail_base"
    )

    # 3. Robust median scale ratio
    valid_lens = (len_top > 10) & (len_bot > 10)
    scale = np.nanmedian(len_top[valid_lens]) / max(np.nanmedian(len_bot[valid_lens]), 1e-4)

    # 4. Translocate BotCam paws into TopCam body-centered virtual arena space
    fused_dict = {}
    unified_scorer = "MultiView_DorsalVentral"

    # Copy TopCam bodyparts
    for bp in TOPCAM_BPS:
        if bp in df_top[scorer_top].columns.levels[0]:
            for coord in ['x', 'y', 'likelihood']:
                vals = df_top[scorer_top][bp][coord].values.copy()
                if coord == 'likelihood':
                    vals = np.clip(vals, 0.0, 1.0)
                fused_dict[(unified_scorer, bp, coord)] = vals

    # Transform ventral paws
    for paw in BOTCAM_PAWS:
        if paw in df_bot[scorer_bot].columns.levels[0]:
            paw_x = df_bot[scorer_bot][paw]['x'].values
            paw_y = df_bot[scorer_bot][paw]['y'].values
            paw_p = np.clip(df_bot[scorer_bot][paw]['likelihood'].values, 0.0, 1.0)

            # Center relative to BotCam centroid
            paw_pts = np.stack([paw_x, paw_y], axis=-1) - c_bot

            # Rotate to egocentric frame (align snout to +x)
            cos_b, sin_b = np.cos(-theta_bot), np.sin(-theta_bot)
            ego_x = (cos_b * paw_pts[:, 0] - sin_b * paw_pts[:, 1]) * scale
            ego_y = (sin_b * paw_pts[:, 0] + cos_b * paw_pts[:, 1]) * scale

            # Project into TopCam virtual arena space via TopCam heading and centroid:
            cos_t, sin_t = np.cos(theta_top), np.sin(theta_top)
            virt_x = c_top[:, 0] + (cos_t * ego_x - sin_t * ego_y)
            virt_y = c_top[:, 1] + (sin_t * ego_x + cos_t * ego_y)

            fused_dict[(unified_scorer, paw, 'x')] = virt_x
            fused_dict[(unified_scorer, paw, 'y')] = virt_y
            fused_dict[(unified_scorer, paw, 'likelihood')] = paw_p

    # Build MultiIndex DataFrame
    columns = pd.MultiIndex.from_tuples(
        list(fused_dict.keys()),
        names=['scorer', 'bodyparts', 'coords']
    )
    df_fused = pd.DataFrame(fused_dict, columns=columns)

    out_path = Path(out_h5_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_fused.to_hdf(str(out_path), key="df_with_missing", mode="w")

    return df_fused, scale, sync_stats

if __name__ == "__main__":
    print("02_fuse_dorsal_ventral module ready.")
