#!/usr/bin/env python3
"""
tests/test_camera_correspondence_synthetic.py
=============================================
Synthetic unit test suite for continuous camera correspondence,
calibration provenance, clock drift recovery, dropped frames,
and strict validation mode.

Runs in < 1 second purely on CPU using isolated temporary fixtures.
"""

import sys
import unittest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from core.camera_correspondence import compute_session_camera_mapping


class TestCameraCorrespondenceSynthetic(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_synthetic_crystal_calibration_success(self):
        """Verifies precise crystal clock calibration and 1-to-1 frame correspondence."""
        n_frames = 300
        dt_ns = 16_666_667  # ~60 Hz in nanoseconds
        true_slope = 1.000018
        true_intercept = 42_000_000.0  # 42 ms offset in ns
        
        b_bas = np.arange(n_frames, dtype=np.float64) * dt_ns + 100_000_000.0
        t_bas = true_slope * b_bas + true_intercept
        
        t_bon = (t_bas - t_bas[0]) / 1e9 + 1000.0
        b_bon = (b_bas - b_bas[0]) / 1e9 + 1000.0
        
        top_csv = self.tmp_path / "top_timestamps.csv"
        bot_csv = self.tmp_path / "bot_timestamps.csv"
        
        pd.DataFrame({
            "Cam_Basler_Timestamp": t_bas,
            "Cam_Bonsai_Timestamp": t_bon,
        }).to_csv(top_csv, index=False)
        
        pd.DataFrame({
            "Cam_Basler_Timestamp": b_bas,
            "Cam_Bonsai_Timestamp": b_bon,
        }).to_csv(bot_csv, index=False)
        
        mapping, dt_ms, calib = compute_session_camera_mapping(
            top_csv, bot_csv, max_jitter_ms=8.35, strict=True
        )
        
        # Provenance assertions
        self.assertEqual(calib["status"], "VALID_CRYSTAL")
        self.assertEqual(calib["mode"], "rate_aware_basler_crystal")
        self.assertTrue(calib["is_valid_crystal"])
        self.assertTrue(calib["strict_validated"])
        self.assertGreaterEqual(calib["n_anchors_retained"], 5)
        
        # Regression coefficient recovery
        np.testing.assert_allclose(calib["slope"], true_slope, rtol=1e-5)
        np.testing.assert_allclose(calib["intercept"], true_intercept, atol=1e5)  # within 0.1 ms
        
        # Mapping verification: 1-to-1 correspondence
        self.assertEqual(len(mapping), n_frames)
        self.assertTrue(np.all(mapping == np.arange(n_frames)))
        self.assertTrue(np.all(dt_ms < 0.5))  # residual jitter < 0.5 ms

    def test_synthetic_dropped_frames(self):
        """Verifies that dropped frames in BotCam are detected and mapped to -1."""
        dt_ns = 16_666_667
        n_frames = 200
        
        b_bas = np.arange(n_frames, dtype=np.float64) * dt_ns
        t_bas = b_bas.copy()
        
        # Simulate a 5-frame drop in BotCam between frame 50 and 55
        drop_mask = np.ones(n_frames, dtype=bool)
        drop_mask[50:55] = False
        b_bas_dropped = b_bas[drop_mask]
        
        t_bon = t_bas / 1e9 + 1000.0
        b_bon_dropped = b_bas_dropped / 1e9 + 1000.0
        
        top_csv = self.tmp_path / "top_drop.csv"
        bot_csv = self.tmp_path / "bot_drop.csv"
        
        pd.DataFrame({"Cam_Basler_Timestamp": t_bas, "Cam_Bonsai_Timestamp": t_bon}).to_csv(top_csv, index=False)
        pd.DataFrame({"Cam_Basler_Timestamp": b_bas_dropped, "Cam_Bonsai_Timestamp": b_bon_dropped}).to_csv(bot_csv, index=False)
        
        mapping, dt_ms, calib = compute_session_camera_mapping(
            top_csv, bot_csv, max_jitter_ms=5.0, strict=False
        )
        
        self.assertEqual(calib["status"], "VALID_CRYSTAL")
        # Middle of dropped frame gap has delta > 5.0ms, must map to -1
        self.assertEqual(mapping[52], -1)

    def test_synthetic_insufficient_anchors_strict_vs_permissive(self):
        """Verifies that <5 anchors report DEGRADED in permissive mode and raise in strict mode."""
        t_bas = np.arange(5, dtype=np.float64) * 16_666_667
        b_bas = t_bas + 10_000_000
        t_bon = t_bas / 1e9
        b_bon = b_bas / 1e9
        
        top_csv = self.tmp_path / "top_short.csv"
        bot_csv = self.tmp_path / "bot_short.csv"
        pd.DataFrame({"Cam_Basler_Timestamp": t_bas, "Cam_Bonsai_Timestamp": t_bon}).to_csv(top_csv, index=False)
        pd.DataFrame({"Cam_Basler_Timestamp": b_bas, "Cam_Bonsai_Timestamp": b_bon}).to_csv(bot_csv, index=False)
        
        # Permissive mode: should return degraded status
        mapping, dt_ms, calib = compute_session_camera_mapping(top_csv, bot_csv, strict=False)
        self.assertEqual(calib["status"], "DEGRADED_INSUFFICIENT_ANCHORS")
        self.assertEqual(calib["mode"], "insufficient_anchors_fallback")
        self.assertFalse(calib["is_valid_crystal"])
        self.assertEqual(calib["slope"], 1.0)
        
        # Strict mode: must raise RuntimeError
        with self.assertRaises(RuntimeError):
            compute_session_camera_mapping(top_csv, bot_csv, strict=True)

    def test_synthetic_missing_basler_columns_strict_vs_permissive(self):
        """Verifies that missing Basler column triggers host fallback in permissive mode and raises in strict mode."""
        t_bon = np.arange(50, dtype=np.float64) * 0.016
        b_bon = t_bon.copy()
        
        top_csv = self.tmp_path / "top_nobasler.csv"
        bot_csv = self.tmp_path / "bot_nobasler.csv"
        pd.DataFrame({"Cam_Bonsai_Timestamp": t_bon}).to_csv(top_csv, index=False)
        pd.DataFrame({"Cam_Bonsai_Timestamp": b_bon}).to_csv(bot_csv, index=False)
        
        # Permissive mode: fallback to host timestamps
        mapping, dt_ms, calib = compute_session_camera_mapping(top_csv, bot_csv, strict=False)
        self.assertEqual(calib["status"], "FALLBACK_HOST_TIMESTAMPS")
        self.assertEqual(calib["mode"], "bonsai_host_fallback")
        self.assertFalse(calib["is_valid_crystal"])
        
        # Strict mode: must raise RuntimeError
        with self.assertRaises(RuntimeError):
            compute_session_camera_mapping(top_csv, bot_csv, strict=True)

    def test_synthetic_malformed_timestamps_validation(self):
        """Verifies that non-monotonic, non-finite, or empty timestamps raise appropriate errors."""
        top_csv = self.tmp_path / "top_bad.csv"
        bot_csv = self.tmp_path / "bot_bad.csv"
        
        # 1. Non-monotonic crystal clock
        pd.DataFrame({
            "Cam_Basler_Timestamp": [100.0, 200.0, 150.0],
            "Cam_Bonsai_Timestamp": [1.0, 2.0, 3.0],
        }).to_csv(top_csv, index=False)
        pd.DataFrame({
            "Cam_Basler_Timestamp": [100.0, 200.0, 300.0],
            "Cam_Bonsai_Timestamp": [1.0, 2.0, 3.0],
        }).to_csv(bot_csv, index=False)
        
        with self.assertRaises(ValueError):
            compute_session_camera_mapping(top_csv, bot_csv)
            
        # 2. Non-finite (NaN)
        pd.DataFrame({
            "Cam_Basler_Timestamp": [100.0, np.nan, 300.0],
            "Cam_Bonsai_Timestamp": [1.0, 2.0, 3.0],
        }).to_csv(top_csv, index=False)
        
        with self.assertRaises(ValueError):
            compute_session_camera_mapping(top_csv, bot_csv)
            
        # 3. Empty dataframe
        pd.DataFrame({"Cam_Basler_Timestamp": [], "Cam_Bonsai_Timestamp": []}).to_csv(top_csv, index=False)
        with self.assertRaises(ValueError):
            compute_session_camera_mapping(top_csv, bot_csv)


if __name__ == "__main__":
    unittest.main()
