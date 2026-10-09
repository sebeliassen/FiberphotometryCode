#!/usr/bin/env python3
"""
fuse_all_tandem_cohort.py
=============================================================================
Batch Multi-Processing Fusion for ALL Available Tandem Sessions (122 Sessions).

Fuses TopCam dorsal tracking (17 keypoints) with BotCam ventral tracking (6 keypoints)
via canonical egocentric body-frame transformation applying the 4 Golden Rules:
1. Fixed robust scale per session from reliable frames.
2. Pre-heading horizontal reflection (x -> 972 - x) on BotCam with bilateral paw swap.
3. Anatomical nose-tail midpoint and heading direction in both views.
4. Circular smoothing on reference headings & median smoothing on centroids;
   paws remain 100% raw, un-smoothed, and unadulterated.

Parallelized across 16 worker processes for maximum throughput.
=============================================================================
"""

import os
import sys
import re
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
TOP_H5_DIR = REPO_ROOT / "karen_project/h5_files"
BOT_PROD_DIR = Path("/home/fsp585/Desktop/production_analysis_arm_B_seed_123_20260922/predictions")
OUT_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/fused_h5_full_cohort"

CHAMBER_MAP = {
    "Arena1": "top_left",
    "Arena2": "bottom_left",
    "Arena3": "bottom_right"
}

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
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.empty((len(angle), 2, 2), dtype=np.float32)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    return R

def smooth_heading_circular(theta: np.ndarray, window: int = 11) -> np.ndarray:
    sin_s = median_filter(np.sin(theta), size=window, mode="nearest")
    cos_s = median_filter(np.cos(theta), size=window, mode="nearest")
    return np.arctan2(sin_s, cos_s)

def smooth_centroid_median(c: np.ndarray, window: int = 11) -> np.ndarray:
    return np.column_stack([
        median_filter(c[:, 0], size=window, mode="nearest"),
        median_filter(c[:, 1], size=window, mode="nearest")
    ])

def fuse_single_pair(pair_info: dict, arena_width: float = 972.0) -> dict:
    sid = pair_info["sid"]
    top_h5_path = pair_info["top_h5"]
    bot_h5_path = pair_info["bot_h5"]
    out_h5_path = OUT_DIR / f"{sid}_tandem_23kp.h5"

    t0 = time.time()
    if out_h5_path.exists() and out_h5_path.stat().st_size > 1_000_000:
        return {"sid": sid, "status": "CACHED", "frames": 0, "time": 0.0, "path": str(out_h5_path)}

    try:
        df_top = pd.read_hdf(top_h5_path)
        df_bot = pd.read_hdf(bot_h5_path)

        top_scorer = df_top.columns.levels[0][0]
        bot_scorer = df_bot.columns.levels[0][0]

        n_frames = min(len(df_top), len(df_bot))
        active_slice = slice(0, n_frames)

        # 1. BotCam landmarks & horizontal mirror reflection
        bot_raw_coords = {}
        for bp in ["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw", "nose_tip", "tail_base"]:
            bx = df_bot[bot_scorer, bp, "x"].iloc[active_slice].to_numpy(dtype=np.float32)
            by = df_bot[bot_scorer, bp, "y"].iloc[active_slice].to_numpy(dtype=np.float32)
            blike = df_bot[bot_scorer, bp, "likelihood"].iloc[active_slice].to_numpy(dtype=np.float32)
            bot_raw_coords[bp] = {"x": arena_width - bx, "y": by, "p": blike}

        # Bilateral swap
        bot_corr = {
            "left_front_paw": bot_raw_coords["right_front_paw"],
            "right_front_paw": bot_raw_coords["left_front_paw"],
            "left_hind_paw": bot_raw_coords["right_hind_paw"],
            "right_hind_paw": bot_raw_coords["left_hind_paw"],
            "nose_tip": bot_raw_coords["nose_tip"],
            "tail_base": bot_raw_coords["tail_base"]
        }

        # 2. TopCam landmarks
        top_nose_x = df_top[top_scorer, "nose", "x"].iloc[active_slice].to_numpy(dtype=np.float32)
        top_nose_y = df_top[top_scorer, "nose", "y"].iloc[active_slice].to_numpy(dtype=np.float32)
        top_tail_x = df_top[top_scorer, "tail_base", "x"].iloc[active_slice].to_numpy(dtype=np.float32)
        top_tail_y = df_top[top_scorer, "tail_base", "y"].iloc[active_slice].to_numpy(dtype=np.float32)

        bot_nose_x = bot_corr["nose_tip"]["x"]
        bot_nose_y = bot_corr["nose_tip"]["y"]
        bot_tail_x = bot_corr["tail_base"]["x"]
        bot_tail_y = bot_corr["tail_base"]["y"]

        # 3. Robust scale
        len_top = np.sqrt((top_nose_x - top_tail_x)**2 + (top_nose_y - top_tail_y)**2)
        len_bot = np.sqrt((bot_nose_x - bot_tail_x)**2 + (bot_nose_y - bot_tail_y)**2)
        reliable = (len_top > 20.0) & (len_bot > 40.0)
        if np.sum(reliable) > 500:
            fixed_scale = float(np.nanmedian(len_top[reliable]) / np.nanmedian(len_bot[reliable]))
        else:
            fixed_scale = 0.3716

        # 4. Centroids and headings
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
        fused_dict = {}
        fused_scorer = "fused_multiview"

        for bp in TOP_USE_BODYPARTS:
            fused_dict[(fused_scorer, bp, "x")] = df_top[top_scorer, bp, "x"].iloc[active_slice].to_numpy(dtype=np.float32)
            fused_dict[(fused_scorer, bp, "y")] = df_top[top_scorer, bp, "y"].iloc[active_slice].to_numpy(dtype=np.float32)
            fused_dict[(fused_scorer, bp, "likelihood")] = df_top[top_scorer, bp, "likelihood"].iloc[active_slice].to_numpy(dtype=np.float32)

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

            fused_dict[(fused_scorer, new_bp, "x")] = proj_x.astype(np.float32)
            fused_dict[(fused_scorer, new_bp, "y")] = proj_y.astype(np.float32)
            fused_dict[(fused_scorer, new_bp, "likelihood")] = bp_like.astype(np.float32)

        columns = pd.MultiIndex.from_tuples(list(fused_dict.keys()), names=["scorer", "bodyparts", "coords"])
        df_fused = pd.DataFrame(fused_dict, index=np.arange(n_frames))
        df_fused.columns = columns

        df_fused.to_hdf(str(out_h5_path), key="df_with_missing", mode="w", format="fixed")
        dur = time.time() - t0
        return {"sid": sid, "status": "FUSED", "frames": n_frames, "time": dur, "path": str(out_h5_path), "scale": fixed_scale}

    except Exception as e:
        return {"sid": sid, "status": "ERROR", "error": str(e), "time": time.time() - t0}

def find_matched_pairs():
    top_files = sorted(TOP_H5_DIR.glob("*.h5"))
    bot_dirs = sorted([d for d in BOT_PROD_DIR.iterdir() if d.is_dir()])
    ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})")

    pairs = []
    for bdir in bot_dirs:
        h5_candidates = list(bdir.glob("*snapshot_best-180.h5")) or list(bdir.glob("*.h5"))
        if not h5_candidates:
            continue
        bot_h5 = h5_candidates[0]

        m_ts = ts_pattern.search(bdir.name)
        if not m_ts:
            continue
        ts = m_ts.group(1)

        arena = None
        for a in ["Arena1", "Arena2", "Arena3"]:
            if a in bdir.name:
                arena = a
                break
        if not arena:
            continue
        quadrant = CHAMBER_MAP.get(arena)

        matching_top = None
        for tf in top_files:
            if ts in tf.name and quadrant in tf.name:
                matching_top = tf
                break

        if matching_top:
            sid = f"{ts}_{quadrant}"
            pairs.append({
                "sid": sid,
                "ts": ts,
                "arena": arena,
                "quadrant": quadrant,
                "top_h5": matching_top,
                "bot_h5": bot_h5
            })
    return pairs

def main():
    print("=" * 80)
    print("🚀 BATCH FUSION PIPELINE FOR ALL AVAILABLE TANDEM SESSIONS")
    print(f"TopCam Directory:   {TOP_H5_DIR}")
    print(f"BotCam Directory:   {BOT_PROD_DIR}")
    print(f"Output Directory:   {OUT_DIR}")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = find_matched_pairs()
    print(f"\nFound {len(pairs)} matched TopCam <-> BotCam pairs across Arenas 1, 2, 3.")

    t0_all = time.time()
    success_count = 0
    total_frames = 0
    max_workers = min(16, os.cpu_count() or 4)
    print(f"Launching fusion with {max_workers} parallel workers...\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(fuse_single_pair, p): p for p in pairs}
        for future in as_completed(future_map):
            res = future.result()
            sid = res["sid"]
            if res["status"] in ["FUSED", "CACHED"]:
                success_count += 1
                total_frames += res.get("frames", 0)
                tag = "⚡ FUSED" if res["status"] == "FUSED" else "💾 CACHED"
                scale_str = f"s={res.get('scale', 0.37):.4f}" if "scale" in res else ""
                print(f"  [{success_count:3d}/{len(pairs)}] {tag} {sid:30s} | {res.get('frames', 0):,d} frames ({res['time']:.1f}s) {scale_str}", flush=True)
            else:
                print(f"  [ERROR] {sid:30s} | {res.get('error', 'unknown error')}", flush=True)

    elapsed = time.time() - t0_all
    print("\n" + "=" * 80)
    print(f"🎉 BATCH FUSION COMPLETE in {elapsed:.1f} seconds!")
    print(f"   Successfully fused: {success_count} / {len(pairs)} sessions")
    print(f"   Total frames:       {total_frames:,} frames across 23 keypoints")
    print(f"   Output folder:      {OUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
