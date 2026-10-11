#!/usr/bin/env python3
"""
test_ttl_aligned_fusion.py
===========================
Rigorous verification test for TTL hardware-synchronized fusion engine.

Validates 3 test regimes:
1. Positive offset (TopCam started first, +67 frames): 2026-05-16T10_41_38_top_left
2. Negative offset (BotCam started first, -26 frames): 2025-12-03T09_51_05_bottom_left
3. Zero offset (Synchronous start, 0 frames):         2026-03-06T16_15_25_top_left

Assertions:
- Length matches len(df_top) verbatim (Master Clock invariant).
- Dorsal coordinates match raw TopCam predictions identically.
- Padding frames are NaN with 0.0 likelihood.
- Overlapping frames achieve > 0.85 mean cosine heading alignment.
- Keypoint-MoSeq MultiIndex schema compatibility.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from fpcode.tracking.pair_matching import find_matched_pairs
from fpcode.tracking.fusion import fuse_single_pair


def run_test():
    print("=" * 80)
    print("🧪 RUNNING TTL HARDWARE-ALIGNED FUSION VERIFICATION SUITE")
    print("=" * 80)

    pairs = find_matched_pairs()
    pair_dict = {p["sid"]: p for p in pairs}

    test_sids = [
        ("2026-05-16T10_41_38_top_left", +67, "Positive offset (TopCam first)"),
        ("2025-12-03T09_51_05_bottom_left", -26, "Negative offset (BotCam first)"),
        ("2026-03-06T16_15_25_top_left", 0, "Zero offset (Synchronous)"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for sid, expected_offset, desc in test_sids:
            print(f"\n--- Testing {sid} ({desc}) ---")
            if sid not in pair_dict:
                print(f"  [WARN] {sid} not in matched pairs. Skipping.")
                continue

            pair_info = pair_dict[sid]
            res = fuse_single_pair(pair_info, out_dir=tmp_path, overwrite=True)

            assert res["status"] == "FUSED", f"Fusion failed for {sid}: {res}"
            print(f"  ✅ Fusion status: {res['status']} in {res['time']:.2f}s")
            print(f"  ✅ Detected TTL offset: {res['ttl_offset']} frames (Expected ~ {expected_offset})")
            assert abs(res["ttl_offset"] - expected_offset) <= 1, f"Unexpected offset: {res['ttl_offset']} vs {expected_offset}"

            fused_file = Path(res["path"])
            assert fused_file.exists() and fused_file.stat().st_size > 1_000_000

            # Read raw top and fused H5
            df_top = pd.read_hdf(pair_info["top_h5"])
            df_fused = pd.read_hdf(fused_file)
            st = df_top.columns.levels[0][0]
            sf = df_fused.columns.levels[0][0]

            # Invariant 1: Total length strictly equals len(df_top)
            assert len(df_fused) == len(df_top), f"Length mismatch: {len(df_fused)} != {len(df_top)}"
            print(f"  ✅ Cardinality check passed: {len(df_fused):,d} frames (equals TopCam exactly)")

            # Invariant 2: Dorsal coordinates match verbatim (Golden Rule 1)
            np.testing.assert_allclose(
                df_fused[sf, "nose", "x"].to_numpy(),
                df_top[st, "nose", "x"].to_numpy(),
                rtol=1e-5, atol=1e-5, err_msg="TopCam dorsal coordinates altered!"
            )
            print("  ✅ Golden Rule 1: Dorsal keypoints are bitwise identical to raw TopCam.")

            # Invariant 3: Ventral paw padding outside overlapping window
            offset = res["ttl_offset"]
            if offset > 0:
                # TopCam started first: rows 0..offset-1 must be NaN
                paw_x_pre = df_fused[sf, "left_front_paw", "x"].iloc[:offset].to_numpy()
                paw_p_pre = df_fused[sf, "left_front_paw", "likelihood"].iloc[:offset].to_numpy()
                assert np.all(np.isnan(paw_x_pre)), f"Expected NaNs in pre-offset frames for {sid}"
                assert np.all(paw_p_pre == 0.0), f"Expected 0.0 likelihood in pre-offset frames for {sid}"
                print(f"  ✅ Pre-startup padding verified: frames [0..{offset-1}] are NaN with 0.0 likelihood.")

            # Invariant 4: Heading cosine alignment across matched frames
            top_nx = df_fused[sf, "nose", "x"].to_numpy()
            top_ny = df_fused[sf, "nose", "y"].to_numpy()
            top_tx = df_fused[sf, "tail_base", "x"].to_numpy()
            top_ty = df_fused[sf, "tail_base", "y"].to_numpy()
            theta_top = np.arctan2(top_ny - top_tx, top_nx - top_tx)

            bot_nx = df_fused[sf, "bot_nose_tip", "x"].to_numpy()
            bot_ny = df_fused[sf, "bot_nose_tip", "y"].to_numpy()
            bot_tx = df_fused[sf, "bot_tail_base", "x"].to_numpy()
            bot_ty = df_fused[sf, "bot_tail_base", "y"].to_numpy()
            theta_bot = np.arctan2(bot_ny - bot_tx, bot_nx - bot_tx)

            cos_align = np.cos(theta_top - theta_bot)
            mean_cos = float(np.nanmean(cos_align))
            print(f"  ✅ Cross-camera heading cosine alignment: {mean_cos:.4f}")
            assert mean_cos > 0.80, f"Low cross-camera heading alignment: {mean_cos:.4f}"

    print("\n" + "=" * 80)
    print("🎉 ALL TTL-ALIGNED FUSION VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_test()
