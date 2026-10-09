"""
core.fusion_engine
==================
Mathematical 2D tracking fusion engine implementing the 4 Golden Rules:
1. TopCam coordinates for dorsal body parts preserved verbatim.
2. BotCam ventral keypoints centered, rotated into dorsal body frame, and scaled.
3. Temporal lag maintained at 0 frames via continuous rate-aware Basler crystal clock regression.
4. Low confidence and missing frames masked as NaNs without corrupting contiguous segments.
5. Anchor-Gated Confidence: BotCam paw confidence is gated by anchor (nose/tail) validity
   and plausibility (25 <= L <= 350 px, p >= 0.8), preventing unverified projections.
6. Clean Dual-View Truncation: Sessions where BotCam acquisition aborted prematurely
   are truncated to the dual-view acquisition window, preventing phantom missing-ventral states.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

try:
    from .camera_correspondence import get_session_mapping
    from .schemas import (
        BOT_RENAME_MAP,
        DEFAULT_ARENA_WIDTH,
        TOP_USE_BODYPARTS,
        PathConfig,
    )
    from .ttl_sync import get_session_ttl_offset
except ImportError:
    from camera_correspondence import get_session_mapping
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
    use_continuous_sync: bool = True,
    confidence_gating: bool = True,
    truncate_aborted_tail: bool = True,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Fuses a matched TopCam and BotCam session into a unified 23-keypoint HDF5 tracking file.
    Supports continuous hardware rate-aware camera mapping and anchor-gated confidence.
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

        # 0. Camera Alignment Mapping
        mapping = None
        if use_continuous_sync:
            mapping = get_session_mapping(sid)

        if mapping is not None:
            N_map = min(N_top, len(mapping))
            mapping = mapping[:N_map]

            # Truncate terminal aborted recordings (e.g. BotCam died early)
            if truncate_aborted_tail:
                valid_mapped = np.where((mapping >= 0) & (mapping < N_bot))[0]
                if len(valid_mapped) > 0 and valid_mapped.max() < N_top - 500:
                    N_top = int(valid_mapped.max() + 1)
                    df_top = df_top.iloc[:N_top]
                    mapping = mapping[:N_top]

            top_idx = np.where((mapping >= 0) & (mapping < N_bot))[0]
            bot_idx = mapping[top_idx]
            sync_mode = "continuous_calibrated"
        else:
            # Fallback to static Master Clock TTL offset
            offset_info = get_session_ttl_offset(sid)
            offset = int(offset_info.get("bot0_in_top", 0) - offset_info.get("top0_in_bot", 0))
            t_start = max(0, offset)
            t_end = min(N_top, N_bot + offset)
            if truncate_aborted_tail and t_end < N_top - 500:
                N_top = t_end
                df_top = df_top.iloc[:N_top]

            top_idx = np.arange(t_start, min(t_end, N_top))
            bot_idx = top_idx - offset
            sync_mode = "static_ttl"

        if len(top_idx) == 0:
            raise ValueError(f"No temporal overlap for {sid}: N_top={N_top}, N_bot={N_bot}")

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

        bot_nose_p = bot_corr["nose_tip"]["p"]
        bot_tail_p = bot_corr["tail_base"]["p"]

        # 3. Robust inter-camera scaling factor
        len_top = np.sqrt((top_nose_x - top_tail_x)**2 + (top_nose_y - top_tail_y)**2)
        len_bot = np.sqrt((bot_nose_x - bot_tail_x)**2 + (bot_nose_y - bot_tail_y)**2)

        # 4. Anatomical Anchor Confidence Gating
        if confidence_gating:
            valid_anchors = (
                (bot_nose_p >= 0.8) &
                (bot_tail_p >= 0.8) &
                (len_bot >= 25.0) &
                (len_bot <= 350.0)
            )
            # Reliable frames for scaling factor must have valid anchors
            reliable = (len_top > 20.0) & (len_bot > 40.0) & valid_anchors
            if np.sum(reliable) > 500:
                fixed_scale = float(np.nanmedian(len_top[reliable]) / np.nanmedian(len_bot[reliable]))
            else:
                fixed_scale = 0.3716

            # Compute raw centroids and headings
            c_top_raw = np.column_stack([0.5 * (top_nose_x + top_tail_x), 0.5 * (top_nose_y + top_tail_y)])
            c_bot_raw = np.column_stack([0.5 * (bot_nose_x + bot_tail_x), 0.5 * (bot_nose_y + bot_tail_y)])

            theta_top_raw = np.arctan2(top_nose_y - top_tail_y, top_nose_x - top_tail_x)
            theta_bot_raw = np.arctan2(bot_nose_y - bot_tail_y, bot_nose_x - bot_tail_x)

            # Local bounded search / interpolation for invalid anchor glitches (<= 10 frames)
            valid_indices = np.where(valid_anchors)[0]
            is_frame_valid = np.zeros(len(top_idx), dtype=bool)

            if len(valid_indices) > 0:
                idx_all = np.arange(len(top_idx))
                nearest_pos = np.searchsorted(valid_indices, idx_all)
                nearest_pos = np.clip(nearest_pos, 0, len(valid_indices) - 1)
                left_pos = np.maximum(0, nearest_pos - 1)

                dist_right = np.abs(valid_indices[nearest_pos] - idx_all)
                dist_left = np.abs(valid_indices[left_pos] - idx_all)
                min_dist = np.minimum(dist_right, dist_left)

                is_frame_valid = (min_dist <= 10)

                # Interpolate c_bot and theta_bot where anchor is temporarily glitching
                invalid_mask = ~valid_anchors
                if np.any(invalid_mask) and len(valid_indices) >= 2:
                    c_bot_raw[invalid_mask, 0] = np.interp(idx_all[invalid_mask], valid_indices, c_bot_raw[valid_indices, 0])
                    c_bot_raw[invalid_mask, 1] = np.interp(idx_all[invalid_mask], valid_indices, c_bot_raw[valid_indices, 1])

                    theta_unwrapped = np.unwrap(theta_bot_raw[valid_indices])
                    theta_interp = np.interp(idx_all[invalid_mask], valid_indices, theta_unwrapped)
                    theta_bot_raw[invalid_mask] = np.mod(theta_interp + np.pi, 2 * np.pi) - np.pi

            theta_top = smooth_heading_circular(theta_top_raw, window=11)
            theta_bot = smooth_heading_circular(theta_bot_raw, window=11)
            c_top = smooth_centroid_median(c_top_raw, window=11)
            c_bot = smooth_centroid_median(c_bot_raw, window=11)

            # Anchor confidence propagates to paws: a paw cannot have higher confidence than its coordinate frame
            anchor_confidence = np.minimum(bot_nose_p, bot_tail_p)
        else:
            reliable = (len_top > 20.0) & (len_bot > 40.0)
            if np.sum(reliable) > 500:
                fixed_scale = float(np.nanmedian(len_top[reliable]) / np.nanmedian(len_bot[reliable]))
            else:
                fixed_scale = 0.3716

            c_top_raw = np.column_stack([0.5 * (top_nose_x + top_tail_x), 0.5 * (top_nose_y + top_tail_y)])
            c_bot_raw = np.column_stack([0.5 * (bot_nose_x + bot_tail_x), 0.5 * (bot_nose_y + bot_tail_y)])

            theta_top_raw = np.arctan2(top_nose_y - top_tail_y, top_nose_x - top_tail_x)
            theta_bot_raw = np.arctan2(bot_nose_y - bot_tail_y, bot_nose_x - bot_tail_x)

            theta_top = smooth_heading_circular(theta_top_raw, window=11)
            theta_bot = smooth_heading_circular(theta_bot_raw, window=11)
            c_top = smooth_centroid_median(c_top_raw, window=11)
            c_bot = smooth_centroid_median(c_bot_raw, window=11)

            is_frame_valid = np.ones(len(top_idx), dtype=bool)
            anchor_confidence = np.ones(len(top_idx), dtype=np.float32)

        R_bot_inv = rot2d(-theta_bot)
        R_top = rot2d(theta_top)

        # 5. Assemble MultiIndex Fused DataFrame
        # Output length strictly equals N_top (TopCam Video Frame 0 to N_top - 1)
        fused_dict = {}
        fused_scorer = "fused_multiview"

        # TopCam keypoints preserved verbatim across all N_top frames (Rule 1)
        for bp in TOP_USE_BODYPARTS:
            fused_dict[(fused_scorer, bp, "x")] = df_top[top_scorer, bp, "x"].iloc[:N_top].to_numpy(dtype=np.float32)
            fused_dict[(fused_scorer, bp, "y")] = df_top[top_scorer, bp, "y"].iloc[:N_top].to_numpy(dtype=np.float32)
            fused_dict[(fused_scorer, bp, "likelihood")] = df_top[top_scorer, bp, "likelihood"].iloc[:N_top].to_numpy(dtype=np.float32)

        # BotCam keypoints aligned into mouse heading coordinate system (Rule 2 + Rule 3)
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

            if confidence_gating:
                # Gated paw confidence
                gated_like = np.minimum(bp_like, anchor_confidence)
                # Invalidate untrusted frames
                proj_x[~is_frame_valid] = np.nan
                proj_y[~is_frame_valid] = np.nan
                gated_like[~is_frame_valid] = 0.0
            else:
                gated_like = bp_like

            arr_x = np.full(N_top, np.nan, dtype=np.float32)
            arr_y = np.full(N_top, np.nan, dtype=np.float32)
            arr_like = np.zeros(N_top, dtype=np.float32)

            arr_x[top_idx] = proj_x.astype(np.float32)
            arr_y[top_idx] = proj_y.astype(np.float32)
            arr_like[top_idx] = gated_like.astype(np.float32)

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
            "sync_mode": sync_mode,
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
