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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
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

from syllable_generation.multiview_top_bot_experiments.core.fusion_engine import (
    fuse_single_pair as core_fuse_single_pair,
    rot2d,
    smooth_heading_circular,
    smooth_centroid_median,
)

def fuse_single_pair(pair_info: dict, arena_width: float = 972.0, overwrite: bool = False) -> dict:
    return core_fuse_single_pair(
        pair_info,
        out_dir=OUT_DIR,
        arena_width=arena_width,
        overwrite=overwrite,
    )

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
    import argparse
    parser = argparse.ArgumentParser(description="Batch Multi-Processing Fusion for ALL Available Tandem Sessions")
    parser.add_argument("--overwrite", action="store_true", help="Force re-fusion of existing files")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 BATCH FUSION PIPELINE FOR ALL AVAILABLE TANDEM SESSIONS (TTL HARDWARE-ALIGNED)")
    print(f"TopCam Directory:   {TOP_H5_DIR}")
    print(f"BotCam Directory:   {BOT_PROD_DIR}")
    print(f"Output Directory:   {OUT_DIR}")
    print(f"Overwrite Existing: {args.overwrite}")
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
        future_map = {executor.submit(fuse_single_pair, p, 972.0, args.overwrite): p for p in pairs}
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
