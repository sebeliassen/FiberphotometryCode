#!/usr/bin/env python3
"""
fuse_cohort_static_v2.py
========================
Fuses the entire 118-session cohort using the epoch-aware Occam's razor static affine projection engine.
- Epoch 1 (sessions < 2026-04-01): March calibrated transforms (median held-out residual ~3.1 px).
- Epoch 2 (sessions >= 2026-04-01): April/May calibrated transforms (median held-out residual ~3.66 px).

Saves to: data/cohort_fused_static_21kp_v2/
Uses atomic writing, validation manifests, and multiprocessing for fast parallel fusion.
Includes comprehensive post-fusion pre-flight verification across all 118 sessions.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from core.pair_matching import find_matched_pairs
from core.static_fusion_engine import fuse_single_pair_static, get_session_epoch
from core.cache_manifest import verify_manifest, OK as CACHE_OK


def _fuse_worker(args):
    pair, out_dir, mode, overwrite = args
    return fuse_single_pair_static(
        pair_info=pair,
        out_dir=out_dir,
        keypoint_mode=mode,
        use_epochs=True,
        overwrite=overwrite
    )


def verify_fused_cohort(out_dir: Path, pairs: list, mode: str = "21kp") -> dict:
    """Verifies that every single session received its intended transform and valid data."""
    print("\n" + "=" * 80)
    print("🔬 PRE-FLIGHT VALIDATION: VERIFYING ALL 118 SESSIONS")
    print("=" * 80)

    epoch_counts = {"epoch1_winter_spring_2026": 0, "epoch2_spring_post_shift_2026": 0}
    arena_counts = {}
    errors = []
    total_valid_frames = 0

    with open(EXP_DIR / "core/epoch_arena_transforms.json", "r") as f:
        epoch_ref = json.load(f)

    for p in pairs:
        sid = p["sid"]
        arena = p["arena"]
        ts = p.get("ts", sid[:19])
        expected_epoch = get_session_epoch(ts)
        expected_A = np.array(epoch_ref[expected_epoch][arena]["matrix_2x2"], dtype=np.float32)
        expected_b = np.array(epoch_ref[expected_epoch][arena]["translation_2d"], dtype=np.float32)

        h5_path = out_dir / f"{sid}_tandem_{mode}.h5"
        manifest_path = out_dir / f"{sid}_tandem_{mode}.h5.manifest.json"

        if not h5_path.exists():
            errors.append(f"Missing HDF5 file: {h5_path.name}")
            continue
        if not manifest_path.exists():
            errors.append(f"Missing manifest file: {manifest_path.name}")
            continue

        with open(manifest_path, "r") as mf:
            mdata = json.load(mf)

        # Check transform recorded in manifest
        recorded_epoch = mdata.get("config", {}).get("epoch")
        if recorded_epoch != expected_epoch:
            errors.append(f"{sid}: Epoch mismatch! Recorded '{recorded_epoch}' vs expected '{expected_epoch}'")

        rec_A = np.array(mdata.get("config", {}).get("affine_matrix", []), dtype=np.float32)
        rec_b = np.array(mdata.get("config", {}).get("affine_translation", []), dtype=np.float32)

        if not np.allclose(rec_A, expected_A, atol=1e-5):
            errors.append(f"{sid}: Affine matrix mismatch!")
        if not np.allclose(rec_b, expected_b, atol=1e-4):
            errors.append(f"{sid}: Affine translation mismatch! Expected {expected_b}, got {rec_b}")

        # Check HDF5 structure
        try:
            df = pd.read_hdf(h5_path)
            total_valid_frames += len(df)
            epoch_counts[expected_epoch] += 1
            arena_counts[(expected_epoch, arena)] = arena_counts.get((expected_epoch, arena), 0) + 1
        except Exception as e:
            errors.append(f"{sid}: Could not read HDF5 ({e})")

    report = {
        "total_sessions": len(pairs),
        "verified_sessions": len(pairs) - len(errors),
        "epoch_counts": epoch_counts,
        "arena_counts": {f"{k[0]}_{k[1]}": v for k, v in arena_counts.items()},
        "total_frames": total_valid_frames,
        "errors": errors,
        "passed": (len(errors) == 0 and len(pairs) == 118),
    }

    print(f"  Verified Sessions: {report['verified_sessions']}/{report['total_sessions']}")
    print(f"  Epoch 1 (< 2026-04-01): {epoch_counts['epoch1_winter_spring_2026']} sessions")
    print(f"  Epoch 2 (>= 2026-04-01): {epoch_counts['epoch2_spring_post_shift_2026']} sessions")
    print(f"  Total frames: {total_valid_frames:,}")
    if errors:
        print(f"  ❌ ERRORS FOUND ({len(errors)}):")
        for err in errors[:5]:
            print(f"     - {err}")
    else:
        print(f"  ✅ ALL 118 SESSIONS PASSED PRE-FLIGHT VERIFICATION WITH EXACT EPOCH TRANSFORMS!")

    return report


def main():
    parser = argparse.ArgumentParser(description="Fuse entire 118-session cohort with epoch-aware static registration.")
    parser.add_argument("--mode", choices=["21kp", "23kp"], default="21kp",
                        help="Keypoint mode to fuse (default: 21kp)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing fused files")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: data/cohort_fused_static_<mode>_v2)")
    args = parser.parse_args()

    mode = args.mode
    out_dir = Path(args.out_dir) if args.out_dir else EXP_DIR / f"data/cohort_fused_static_{mode}_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🚀 FUSING FULL COHORT WITH EPOCH-AWARE STATIC REGISTRATION (V2, {mode.upper()})")
    print(f"Destination: {out_dir}")
    print(f"Parallel Workers: {args.workers}")
    print(f"Epoch routing: epoch_arena_transforms.json")
    print("=" * 80)

    pairs = find_matched_pairs()
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
                if res["status"] in ("FUSED", "CACHED", "ok", "cached"):
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
    print(f"🏁 COHORT V2 FUSION COMPLETE: {mode.upper()}")
    print(f"  Success: {success_cnt}/{len(pairs)}")
    print(f"  Failed:  {fail_cnt}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Total time:   {elapsed:.1f}s ({elapsed / max(1, success_cnt):.2f}s per session)")
    print("=" * 80)

    # Pre-flight validation
    report = verify_fused_cohort(out_dir, pairs, mode)
    report_file = out_dir / "fusion_v2_preflight_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved pre-flight verification report to {report_file}")

    if not report["passed"]:
        print("❌ CRITICAL: Cohort validation failed! Aborting.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
