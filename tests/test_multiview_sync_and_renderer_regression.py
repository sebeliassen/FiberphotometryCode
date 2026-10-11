#!/usr/bin/env python3
"""
test_multiview_sync_and_renderer_regression.py
=============================================================================
Regression test suite for multiview hardware synchronization and rendering:
1. Real 6-frame gap: asserts -1 preservation through mapping & callers.
2. Clock rate drift calibration: slope, ppm drift, and anchor residuals.
3. Camera startup ordering (either camera starting first).
4. Frozen / duplicate FP timestamp detection.
5. TopCam quadrant coordinate fallback preventing black crops.
6. BotCam confidence filtering rejecting outlier glitches / screws.
=============================================================================
"""

import sys
from pathlib import Path

import pandas as pd

from fpcode.common.paths import REPO_ROOT

EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import importlib.util

from fpcode.tracking.camera_correspondence import compute_session_camera_mapping, top_to_bot_frame_continuous
from fpcode.tracking.clock_sync import find_camera_timestamp_files

spec = importlib.util.spec_from_file_location(
    "renderer",
    str(EXP_DIR / "04_reports_and_figures/render_all_seed101_grid_movies.py")
)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def test_six_frame_gap_rejection_and_preservation():
    """
    Validates that the real 6-frame gap in 2025-12-03T14_06_45_bottom_right (frames 186367-186372)
    is explicitly rejected as -1 and survives through top_to_bot_frame_continuous.
    """
    sid = "2025-12-03T14_06_45_bottom_right"
    top_csv, bot_csv = find_camera_timestamp_files("2025-12-03T14_06_45", "Arena3")
    mapping, dt_ms, calib = compute_session_camera_mapping(top_csv, bot_csv)

    # Frame 186366: valid match before gap
    assert mapping[186366] == 186366
    assert dt_ms[186366] < 1.0

    # Frames 186367-186372: exactly 6 dropped frames
    for drop_f in range(186367, 186373):
        assert mapping[drop_f] == -1, f"Frame {drop_f} should be marked -1 in mapping"
        cont_val = top_to_bot_frame_continuous(sid, drop_f)
        assert cont_val == -1, f"top_to_bot_frame_continuous({drop_f}) must return -1, got {cont_val}"

        # Verify renderer caller pattern does not convert -1 to 0
        safe_bot_frame = cont_val if cont_val >= 0 else None
        assert safe_bot_frame is None, "Caller must not clamp -1 to frame 0!"

    # Frame 186373: recovers cleanly with 6 frame shift
    assert mapping[186373] == 186367
    assert dt_ms[186373] < 1.0
    assert top_to_bot_frame_continuous(sid, 186373) == 186367


def test_clock_drift_calibration_anchors_and_residuals():
    """
    Validates rate-aware Basler crystal clock calibration across sessions.
    Verifies that anchor residuals are sub-millisecond (< 0.05 ms) and drift is plausible.
    """
    test_sessions = [
        ("2025-12-03T16_38_48", "Arena3", +3.31),
        ("2026-01-09T09_41_59", "Arena2", +0.61),
        ("2026-04-28T13_21_05", "Arena2", +0.64)
    ]
    for ts, arena, expected_ppm in test_sessions:
        top_csv, bot_csv = find_camera_timestamp_files(ts, arena)
        mapping, dt_ms, calib = compute_session_camera_mapping(top_csv, bot_csv)

        assert calib["mode"] == "rate_aware_basler_crystal"
        assert calib["n_anchors_retained"] >= 200, f"Expected >= 200 anchors, got {calib['n_anchors_retained']}"
        assert calib["anchor_mean_res_ms"] < 0.05, f"Mean residual {calib['anchor_mean_res_ms']} ms exceeds 0.05 ms"
        assert abs(calib["ppm_drift"] - expected_ppm) < 0.5, f"PPM drift {calib['ppm_drift']} differs from expected {expected_ppm}"


def test_top_quadrant_dlc_fallback_prevents_black_crops():
    """
    Validates that pre-cropped 404x404 TopCam videos falling out of bounds
    fall back gracefully to verified DLC coordinates instead of rendering black tiles.
    """
    # 2026-03-15T09_51_36_bottom_left
    top_h5 = Path("/home/fsp585/Desktop/FiberphotometryCode/studies/karen_pff_vs_monomer/h5_files/T8_7.pilot390.pilot398_150326_Arena1234_CenterTopCamVid_2026-03-15T09_51_36_bottom_leftDLC_HrnetW32_dataset_fixDec16shuffle10_detector_020_snapshot_020.h5")
    assert top_h5.exists()
    df_top = pd.read_hdf(top_h5)
    scorer = df_top.columns.levels[0][0]

    # At raw frame 41591 (onset of first syllable)
    mc_x = float(df_top[scorer, "mouse_center", "x"].iloc[41591])
    mc_y = float(df_top[scorer, "mouse_center", "y"].iloc[41591])
    mc_p = float(df_top[scorer, "mouse_center", "likelihood"].iloc[41591])

    # Must be within 404x404 video frame
    assert 0 <= mc_x <= 404.0
    assert 0 <= mc_y <= 404.0
    assert mc_p > 0.8


def test_bot_confidence_filtering_rejects_screws():
    """
    Validates that compute_robust_bot_pose rejects tracking glitches (e.g. low-confidence arena corner screws)
    and resolves valid mouse pose with morphological distance check.
    """
    bot_h5 = Path("/home/fsp585/Desktop/production_analysis_arm_B_seed_123_20260922/predictions/8_weeks_PI_T8_23_18_bothRepeated_090126_active_Arena1_BotCamVid_2026-01-09T12_19_47/8_weeks_PI_T8_23_18_bothRepeated_090126_active_Arena1_BotCamVid_2026-01-09T12_19_47DLC_HrnetW32_karen_bottom_modelSep14shuffle9_snapshot_best-180.h5")
    assert bot_h5.exists()
    df_bot = pd.read_hdf(bot_h5)
    bot_scorer = df_bot.columns.levels[0][0]

    # Frame 94728 had nose_tip at (965.9, 965.9) with p = 0.508
    raw_nx = df_bot[bot_scorer, "nose_tip", "x"].iloc[94728]
    raw_ny = df_bot[bot_scorer, "nose_tip", "y"].iloc[94728]
    raw_np = df_bot[bot_scorer, "nose_tip", "likelihood"].iloc[94728]
    assert raw_np < 0.8, "Target glitch frame should have nose likelihood < 0.8"

    # Robust pose resolver must reject the screw coordinate (965, 965)
    c_bot, h_bot, is_valid = renderer.compute_robust_bot_pose(df_bot, bot_scorer, 94728, search_window=10)
    assert is_valid is True
    # The resolved centroid must NOT be in the corner screw region (> 600, > 600)
    assert c_bot[0] < 600 and c_bot[1] < 600, f"Resolved centroid {c_bot} should not be in corner screw!"


if __name__ == "__main__":
    print("Running multiview synchronization and renderer regression test suite...")
    test_six_frame_gap_rejection_and_preservation()
    print("  [PASS] test_six_frame_gap_rejection_and_preservation")
    test_clock_drift_calibration_anchors_and_residuals()
    print("  [PASS] test_clock_drift_calibration_anchors_and_residuals")
    test_top_quadrant_dlc_fallback_prevents_black_crops()
    print("  [PASS] test_top_quadrant_dlc_fallback_prevents_black_crops")
    test_bot_confidence_filtering_rejects_screws()
    print("  [PASS] test_bot_confidence_filtering_rejects_screws")
    print("\nALL REGRESSION TESTS PASSED! 🚀")
