#!/usr/bin/env python3
"""
fuse_pilot_static.py
====================
Fuses the 10 pilot sessions under both:
  - Arm A: 21 keypoints (17 dorsal + 4 paws)
  - Arm B: 23 keypoints (17 dorsal + 4 paws + 2 anchors)
using the Occam's razor static affine projection engine.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from core.pair_matching import find_matched_pairs
from core.static_fusion_engine import fuse_single_pair_static

OUT_DIR_21KP = EXP_DIR / "data/pilot_fused_static_21kp"
OUT_DIR_23KP = EXP_DIR / "data/pilot_fused_static_23kp"

PILOT_SESSIONS = [
    # Arena 1 (4 sessions)
    "2026-03-05T11_01_01_top_left",
    "2026-03-05T12_46_18_top_left",
    "2026-03-06T09_48_02_top_left",
    "2026-03-06T11_14_44_top_left",
    # Arena 2 (3 sessions)
    "2026-03-05T11_01_01_bottom_left",
    "2026-03-05T12_46_18_bottom_left",
    "2026-03-06T09_48_02_bottom_left",
    # Arena 3 (3 sessions)
    "2026-03-05T11_01_01_bottom_right",
    "2026-03-05T12_46_18_bottom_right",
    "2026-03-06T09_48_02_bottom_right",
]

def main():
    print("=" * 80)
    print("⚡ OCCAM'S RAZOR STATIC FUSION: 10 PILOT SESSIONS")
    print(f"Arm A Destination (21kp): {OUT_DIR_21KP}")
    print(f"Arm B Destination (23kp): {OUT_DIR_23KP}")
    print("=" * 80)

    pairs = find_matched_pairs()
    pair_map = {p["sid"]: p for p in pairs}

    missing = [s for s in PILOT_SESSIONS if s not in pair_map]
    if missing:
        print(f"Warning: {len(missing)} sessions not in matched pairs: {missing}")

    pilot_pairs = [pair_map[s] for s in PILOT_SESSIONS if s in pair_map]
    print(f"Found {len(pilot_pairs)} matched pilot pairs to fuse.\n")

    for mode, out_dir in [("21kp", OUT_DIR_21KP), ("23kp", OUT_DIR_23KP)]:
        print(f"\n--- Fusing Mode: {mode.upper()} ---")
        t_start = time.time()
        for i, pair in enumerate(pilot_pairs, 1):
            res = fuse_single_pair_static(
                pair_info=pair,
                out_dir=out_dir,
                keypoint_mode=mode,
                overwrite=True,
            )
            print(f"  [{i:2d}/{len(pilot_pairs)}] {res['status']:6s} | {pair['sid']:32s} | "
                  f"{res.get('frames', 0):6d} frames ({res.get('time', 0):.2f}s)")
        print(f"Completed {mode} fusion in {time.time() - t_start:.2f}s.")

    # Validation
    print("\n" + "=" * 80)
    print("🔍 VALIDATION CHECK OF FUSED HDF5 ARRAYS")
    print("=" * 80)
    for mode, out_dir in [("21kp", OUT_DIR_21KP), ("23kp", OUT_DIR_23KP)]:
        files = list(out_dir.glob("*.h5"))
        print(f"\nMode {mode}: {len(files)} files created in {out_dir.name}")
        sample = pd.read_hdf(files[0])
        bps = sample.columns.levels[1].tolist()
        print(f"  Sample shape: {sample.shape}")
        print(f"  Bodyparts count: {len(bps)} ({bps})")
        # Check non-null percentage for dorsal and paws
        paws = ["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw"]
        paw_nonnull = [sample[("fused_multiview", p, "x")].notna().mean() * 100 for p in paws]
        print(f"  Paw data completeness: {np.mean(paw_nonnull):.1f}% valid frames")

if __name__ == "__main__":
    main()
