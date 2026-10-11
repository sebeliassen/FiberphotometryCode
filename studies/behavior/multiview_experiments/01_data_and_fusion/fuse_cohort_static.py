#!/usr/bin/env python3
"""
fuse_cohort_static.py
=====================
Fuses the entire 118-session cohort using the Occam's razor static affine projection engine.
Supports both 21kp (default, dorsal + 4 paws) and 23kp (dorsal + 4 paws + bot anchors).
Uses atomic writing, validation manifests, and multiprocessing for fast parallel fusion.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from fpcode.common.paths import get_repo_root

REPO_ROOT = get_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from fpcode.tracking.pair_matching import find_matched_pairs
from fpcode.tracking.static_fusion import fuse_single_pair_static
from fpcode.adapters.multiview_compatibility import (
    TTL_CACHE_PATH, legacy_static_calibration, legacy_tracking_paths,
)


def _fuse_worker(args):
    pair, out_dir, mode, overwrite = args
    transform, source, epoch = legacy_static_calibration(pair.get("arena", "Arena1"),
                                                        pair.get("ts", pair["sid"][:19]))
    return fuse_single_pair_static(
        pair_info=pair,
        out_dir=out_dir,
        keypoint_mode=mode,
        overwrite=overwrite,
        ttl_cache_path=TTL_CACHE_PATH,
        transform_info=transform,
        transform_source_path=source,
        epoch=epoch,
    )


def main():
    parser = argparse.ArgumentParser(description="Fuse entire 118-session cohort with static registration.")
    parser.add_argument("--mode", choices=["21kp", "23kp"], default="21kp",
                        help="Keypoint mode to fuse (default: 21kp)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing fused files")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: data/cohort_fused_static_<mode>)")
    args = parser.parse_args()

    mode = args.mode
    out_dir = Path(args.out_dir) if args.out_dir else EXP_DIR / f"data/cohort_fused_static_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🚀 FUSING FULL COHORT WITH STATIC REGISTRATION ({mode.upper()})")
    print(f"Destination: {out_dir}")
    print(f"Parallel Workers: {args.workers}")
    print("=" * 80)

    paths = legacy_tracking_paths()
    pairs = find_matched_pairs(top_h5_dir=paths.top_h5_dir, bot_prod_dir=paths.bot_prod_dir)
    print(f"Found {len(pairs)} matched camera pairs across Arenas 1, 2, and 3.\n")

    t_start = time.time()
    worker_tasks = [(pair, out_dir, mode, args.overwrite) for pair in pairs]

    success_cnt = 0
    fail_cnt = 0
    total_frames = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_fuse_worker, task): task[0]["sid"] for task in worker_tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            sid = futures[fut]
            try:
                res = fut.result()
                if res["status"] in ("FUSED", "CACHED", "CACHED_VALID", "ok", "cached"):
                    success_cnt += 1
                    total_frames += res.get("frames", 0)
                    print(f"  [{i:3d}/{len(pairs)}] {res['status']:6s} | {sid:35s} | "
                          f"{res.get('frames', 0):6d} frames ({res.get('time', 0):.2f}s)")
                else:
                    fail_cnt += 1
                    print(f"  [{i:3d}/{len(pairs)}] FAIL   | {sid:35s} | Error: {res.get('error')}")
            except Exception as e:
                fail_cnt += 1
                print(f"  [{i:3d}/{len(pairs)}] CRASH  | {sid:35s} | Exception: {e}")

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"🏁 COHORT FUSION COMPLETE: {mode.upper()}")
    print(f"  Success: {success_cnt}/{len(pairs)}")
    print(f"  Failed:  {fail_cnt}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Total time:   {elapsed:.1f}s ({elapsed / max(1, success_cnt):.2f}s per session)")
    print("=" * 80)

    # Sanity check
    files = sorted(out_dir.glob("*.h5"))
    print(f"\nVerifying {len(files)} generated HDF5 files in {out_dir}...")
    if files:
        sample = pd.read_hdf(files[0])
        bps = sample.columns.levels[1].tolist()
        print(f"Sample: {files[0].name}")
        print(f"  Shape: {sample.shape}")
        print(f"  Bodyparts ({len(bps)}): {bps}")


if __name__ == "__main__":
    main()
