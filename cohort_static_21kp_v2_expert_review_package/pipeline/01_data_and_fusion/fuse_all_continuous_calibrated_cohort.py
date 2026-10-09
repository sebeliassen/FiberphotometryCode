#!/usr/bin/env python3
"""
fuse_all_continuous_calibrated_cohort.py
=============================================================================
Batch Multi-Processing Continuous Calibrated Fusion for ALL Available Tandem Sessions (118 Sessions).

Enforces:
1. Strict continuous rate-aware Basler crystal clock regression (no static offsets).
2. Dual-anchor plausibility and confidence gating on both TopCam and BotCam nose/tail.
3. Transformed paw confidence strictly capped by anchor confidence.
4. Scale factor estimated strictly on mutual trusted anchor frames.
5. Aborted tail truncation to the dual-view recording window.

Parallelized across worker processes for rapid throughput.
=============================================================================
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from audit_cohort_correlations import find_matched_pairs
from core.fusion_engine import fuse_single_pair

OUT_DIR = EXP_DIR / "fused_h5_continuous_calibrated"

def fuse_worker(pair_info: dict, overwrite: bool = False) -> dict:
    return fuse_single_pair(
        pair_info,
        out_dir=OUT_DIR,
        require_continuous_sync=True,
        confidence_gating=True,
        truncate_aborted_tail=True,
        overwrite=overwrite,
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch Continuous Calibrated Fusion for Tandem Cohort")
    parser.add_argument("--overwrite", action="store_true", help="Force re-fusion of existing files")
    parser.add_argument("--workers", type=int, default=12, help="Number of worker processes")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 BATCH CONTINUOUS-CALIBRATED FUSION (118 SESSIONS)")
    print(f"Output Directory:   {OUT_DIR}")
    print(f"Overwrite Existing: {args.overwrite}")
    print(f"Workers:            {args.workers}")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = find_matched_pairs()
    print(f"\nFound {len(pairs)} matched TopCam <-> BotCam pairs.")

    t0_all = time.time()
    success_count = 0
    total_frames = 0
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(fuse_worker, p, args.overwrite): p for p in pairs}
        for future in as_completed(future_map):
            try:
                res = future.result()
                sid = res["sid"]
                if res["status"] in ["FUSED", "CACHED"]:
                    success_count += 1
                    total_frames += res.get("frames", 0)
                    tag = "⚡ FUSED" if res["status"] == "FUSED" else "💾 CACHED"
                    scale_str = f"s={res.get('scale', 0.37):.4f}" if "scale" in res and not np.isnan(res.get('scale', np.nan)) else ""
                    print(f"  [{success_count:3d}/{len(pairs)}] {tag} {sid:32s} | {res.get('frames', 0):,d} frames ({res['time']:.1f}s) {scale_str}", flush=True)
                else:
                    errors.append((sid, res.get("error", "unknown error")))
                    print(f"  [ERROR] {sid:32s} | {res.get('error', 'unknown error')}", flush=True)
            except Exception as e:
                p = future_map[future]
                errors.append((p['sid'], str(e)))
                print(f"  [EXCEPTION] {p['sid']:32s} | {e}", flush=True)

    elapsed = time.time() - t0_all
    print("\n" + "=" * 80)
    print(f"🎉 BATCH FUSION COMPLETE in {elapsed:.1f} seconds!")
    print(f"   Successfully processed: {success_count} / {len(pairs)} sessions")
    print(f"   Total frames:          {total_frames:,} frames across 23 keypoints")
    print(f"   Errors:                {len(errors)}")
    print(f"   Output folder:         {OUT_DIR}")
    print("=" * 80)

    if errors:
        print("\nErrors encountered:")
        for sid, err in errors:
            print(f"  - {sid}: {err}")

if __name__ == "__main__":
    main()
