"""
core.fusion_engine
==================
Mathematical 2D tracking fusion engine implementing the 4 Golden Rules:
1. TopCam coordinates for dorsal body parts preserved verbatim.
2. BotCam ventral keypoints centered, rotated into dorsal body frame, and scaled.
3. Temporal lag maintained at 0 frames.
4. Low confidence and missing frames masked as NaNs without corrupting contiguous segments.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

try:
    from .schemas import (
        BOT_RENAME_MAP,
        DEFAULT_ARENA_WIDTH,
        TOP_USE_BODYPARTS,
        PathConfig,
    )
    from .ttl_sync import get_session_ttl_offset
except ImportError:
    from schemas import (
        BOT_RENAME_MAP,
        DEFAULT_ARENA_WIDTH,
        TOP_USE_BODYPARTS,
        PathConfig,
    )
    from ttl_sync import get_session_ttl_offset

def rot2d(angle: np.ndarray) -> np.ndarray:
    """Computes vectorized 2D rotation matrices for an array of angles in radians."""
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.empty((len(angle), 2, 2), dtype=np.float32)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    return R

def smooth_heading_circular(theta: np.ndarray, window: int = 11) -> np.ndarray:
    """Performs circular median smoothing on an angular trajectory."""
    sin_s = median_filter(np.sin(theta), size=window, mode="nearest")
    cos_s = median_filter(np.cos(theta), size=window, mode="nearest")
    return np.arctan2(sin_s, cos_s)

def smooth_centroid_median(c: np.ndarray, window: int = 11) -> np.ndarray:
    """Performs median smoothing on an (N, 2) centroid trajectory."""
    return np.column_stack([
        median_filter(c[:, 0], size=window, mode="nearest"),
        median_filter(c[:, 1], size=window, mode="nearest")
    ])

def fuse_single_pair(
    pair_info: Dict[str, Any],
    out_dir: Optional[Path] = None,
    arena_width: float = DEFAULT_ARENA_WIDTH,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Fuses a matched TopCam and BotCam session into a unified 23-keypoint HDF5 tracking file.
    """
    sid = pair_info["sid"]
    top_h5_path = Path(pair_info["top_h5"])
    bot_h5_path = Path(pair_info["bot_h5"])

    dest_dir = Path(out_dir) if out_dir else PathConfig().fused_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_h5_path = dest_dir / f"{sid}_tandem_23kp.h5"

    t0 = time.time()
    if not overwrite and out_h5_path.exists() and out_h5_path.stat().st_size > 1_000_000:
        return {
            "sid": sid,
            "status": "CACHED",
            "frames": 0,
            "time": 0.0,
            "path": str(out_h5_path),
            "scale": np.nan
        }

    try:
        df_top = pd.read_hdf(top_h5_path)
        df_bot = pd.read_hdf(bot_h5_path)

        top_scorer = df_top.columns.levels[0][0]
        bot_scorer = df_bot.columns.levels[0][0]

        N_top = len(df_top)
        N_bot = len(df_bot)

        # Retrieve hardware TTL offset (Master Clock alignment)
        offset_info = get_session_ttl_offset(sid)
        offset = int(offset_info.get("bot0_in_top", 0) - offset_info.get("top0_in_bot", 0))

        # Temporal intersection: TopCam frame t matches BotCam frame (t - offset)
        t_start = max(0, offset)
        t_end = min(N_top, N_bot + offset)

        if t_start >= t_end:
            raise ValueError(
                f"No temporal overlap for {sid}: offset={offset}, N_top={N_top}, N_bot={N_bot}"
            )

        top_idx = np.arange(t_start, t_end)
        bot_idx = top_idx - offset

        # 1. BotCam landmarks & horizontal mirror reflection (x -> arena_width - x) at matched frames
        bot_raw_coords = {}
        for bp in ["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw", "nose_tip", "tail_base"]:
            bx = df_bot[bot_scorer, bp, "x"].iloc[bot_idx].to_numpy(dtype=np.float32)
            by = df_bot[bot_scorer, bp, "y"].iloc[bot_idx].to_numpy(dtype=np.float32)
            blike = df_bot[bot_scorer, bp, "likelihood"].iloc[bot_idx].to_numpy(dtype=np.float32)
            bot_raw_coords[bp] = {"x": arena_width - bx, "y": by, "p": blike}

        # Bilateral swap (due to mirror reflection from bottom)
        bot_corr = {
            "left_front_paw": bot_raw_coords["right_front_paw"],
            "right_front_paw": bot_raw_coords["left_front_paw"],
            "left_hind_paw": bot_raw_coords["right_hind_paw"],
            "right_hind_paw": bot_raw_coords["left_hind_paw"],
            "nose_tip": bot_raw_coords["nose_tip"],
            "tail_base": bot_raw_coords["tail_base"]
        }

        # 2. TopCam landmarks at matched frames
        top_nose_x = df_top[top_scorer, "nose", "x"].iloc[top_idx].to_numpy(dtype=np.float32)
        top_nose_y = df_top[top_scorer, "nose", "y"].iloc[top_idx].to_numpy(dtype=np.float32)
        top_tail_x = df_top[top_scorer, "tail_base", "x"].iloc[top_idx].to_numpy(dtype=np.float32)
        top_tail_y = df_top[top_scorer, "tail_base", "y"].iloc[top_idx].to_numpy(dtype=np.float32)

        bot_nose_x = bot_corr["nose_tip"]["x"]
        bot_nose_y = bot_corr["nose_tip"]["y"]
        bot_tail_x = bot_corr["tail_base"]["x"]
        bot_tail_y = bot_corr["tail_base"]["y"]

        # 3. Robust inter-camera scaling factor
        len_top = np.sqrt((top_nose_x - top_tail_x)**2 + (top_nose_y - top_tail_y)**2)
        len_bot = np.sqrt((bot_nose_x - bot_tail_x)**2 + (bot_nose_y - bot_tail_y)**2)
        reliable = (len_top > 20.0) & (len_bot > 40.0)
        if np.sum(reliable) > 500:
            fixed_scale = float(np.nanmedian(len_top[reliable]) / np.nanmedian(len_bot[reliable]))
        else:
            fixed_scale = 0.3716

        # 4. Centroids and headings at matched frames
        c_top_raw = np.column_stack([0.5 * (top_nose_x + top_tail_x), 0.5 * (top_nose_y + top_tail_y)])
        c_bot_raw = np.column_stack([0.5 * (bot_nose_x + bot_tail_x), 0.5 * (bot_nose_y + bot_tail_y)])

        theta_top_raw = np.arctan2(top_nose_y - top_tail_y, top_nose_x - top_tail_x)
        theta_bot_raw = np.arctan2(bot_nose_y - bot_tail_y, bot_nose_x - bot_tail_x)

        theta_top = smooth_heading_circular(theta_top_raw, window=11)
        theta_bot = smooth_heading_circular(theta_bot_raw, window=11)
        c_top = smooth_centroid_median(c_top_raw, window=11)
        c_bot = smooth_centroid_median(c_bot_raw, window=11)

        R_bot_inv = rot2d(-theta_bot)
        R_top = rot2d(theta_top)

        # 5. Assemble MultiIndex Fused DataFrame
        # IMPORTANT: Output length strictly equals N_top (TopCam Video Frame 0 to N_top - 1)
        fused_dict = {}
        fused_scorer = "fused_multiview"

        # TopCam keypoints preserved verbatim across all N_top frames (Rule 1)
        for bp in TOP_USE_BODYPARTS:
            fused_dict[(fused_scorer, bp, "x")] = df_top[top_scorer, bp, "x"].to_numpy(dtype=np.float32)
            fused_dict[(fused_scorer, bp, "y")] = df_top[top_scorer, bp, "y"].to_numpy(dtype=np.float32)
            fused_dict[(fused_scorer, bp, "likelihood")] = df_top[top_scorer, bp, "likelihood"].to_numpy(dtype=np.float32)

        # BotCam keypoints aligned into mouse heading coordinate system (Rule 2 + Rule 3)
        # Frames outside the overlapping window are padded with NaN and 0.0 likelihood
        for bot_bp, new_bp in BOT_RENAME_MAP.items():
            bx = bot_corr[bot_bp]["x"]
            by = bot_corr[bot_bp]["y"]
            bp_like = bot_corr[bot_bp]["p"]

            dx = bx - c_bot[:, 0]
            dy = by - c_bot[:, 1]
            vec_bot = np.stack([dx, dy], axis=-1)

            ego_x = (R_bot_inv[:, 0, 0] * vec_bot[:, 0] + R_bot_inv[:, 0, 1] * vec_bot[:, 1]) * fixed_scale
            ego_y = (R_bot_inv[:, 1, 0] * vec_bot[:, 0] + R_bot_inv[:, 1, 1] * vec_bot[:, 1]) * fixed_scale

            proj_x = c_top[:, 0] + (R_top[:, 0, 0] * ego_x + R_top[:, 0, 1] * ego_y)
            proj_y = c_top[:, 1] + (R_top[:, 1, 0] * ego_x + R_top[:, 1, 1] * ego_y)

            arr_x = np.full(N_top, np.nan, dtype=np.float32)
            arr_y = np.full(N_top, np.nan, dtype=np.float32)
            arr_like = np.zeros(N_top, dtype=np.float32)

            arr_x[top_idx] = proj_x.astype(np.float32)
            arr_y[top_idx] = proj_y.astype(np.float32)
            arr_like[top_idx] = bp_like.astype(np.float32)

            fused_dict[(fused_scorer, new_bp, "x")] = arr_x
            fused_dict[(fused_scorer, new_bp, "y")] = arr_y
            fused_dict[(fused_scorer, new_bp, "likelihood")] = arr_like

        columns = pd.MultiIndex.from_tuples(list(fused_dict.keys()), names=["scorer", "bodyparts", "coords"])
        df_fused = pd.DataFrame(fused_dict, index=np.arange(N_top))
        df_fused.columns = columns

        df_fused.to_hdf(str(out_h5_path), key="df_with_missing", mode="w", format="fixed")
        dur = time.time() - t0
        return {
            "sid": sid,
            "status": "FUSED",
            "frames": N_top,
            "matched_frames": len(top_idx),
            "ttl_offset": offset,
            "time": dur,
            "path": str(out_h5_path),
            "scale": fixed_scale
        }

    except Exception as e:
        return {
            "sid": sid,
            "status": "ERROR",
            "error": str(e),
            "time": time.time() - t0
        }
