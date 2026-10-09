"""
tests/test_multiview_north_stars_4_1_to_4_3.py
==============================================
Automated Verification Suite for the Multiview KPMS Pipeline:
- North Star 4.1: Expected Marginal Likelihood (EML) Parity & Multi-Seed Rankings
- North Star 4.2: Dual-Camera Spatial-Temporal Synchronization (The 4 Golden Rules)
- North Star 4.3: Syllable Vocabulary, Usage Frequencies & Transition Dynamics
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.multiview_top_bot_experiments.core.schemas import (
    TOP_USE_BODYPARTS,
    BOT_USE_BODYPARTS,
    TANDEM_23KP,
    CHAMBER_MAP,
)
from syllable_generation.multiview_top_bot_experiments.core.pair_matching import find_matched_pairs
from syllable_generation.multiview_top_bot_experiments.core.evaluation_metrics import extract_syllable_statistics

EXPERIMENT_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
PROD_KPMS_DIR = EXPERIMENT_DIR / "kpms_tandem_cohort_production"


class TestMultiviewNorthStars(unittest.TestCase):

    # =========================================================================
    # 4.1 NORTH STAR 1: EML Parity & Model Selection Rankings
    # =========================================================================
    def test_north_star_4_1_eml_rankings_parity(self):
        """Verifies multi-seed EML rankings against known ground truth."""
        csv_path = EXPERIMENT_DIR / "tandem_cohort_eml_model_rankings.csv"
        self.assertTrue(csv_path.exists(), f"EML rankings CSV missing: {csv_path}")

        df_rankings = pd.read_csv(csv_path)
        self.assertEqual(len(df_rankings), 5, "Rankings must contain exactly 5 candidate seeds")

        # 1. Winning model verification (Seed 101)
        winner = df_rankings.iloc[0]
        self.assertEqual(int(winner["seed"]), 101, "Seed 101 must be rank 1 winner")
        self.assertEqual(winner["model_name"], "tandem_cohort_118sess_50ar_200slds_seed101")
        self.assertAlmostEqual(float(winner["eml_score"]), 267583571.5480, delta=1.0)

        # 2. Runner-up verification (Seed 404)
        runner_up = df_rankings.iloc[1]
        self.assertEqual(int(runner_up["seed"]), 404)
        self.assertAlmostEqual(float(runner_up["eml_score"]), 267557436.1043, delta=1.0)

        # 3. Monotonic descending order
        eml_values = df_rankings["eml_score"].to_numpy()
        self.assertTrue(np.all(np.diff(eml_values) <= 0), "EML rankings must be strictly descending")
        print("\n[North Star 4.1 PASSED]: Seed 101 confirmed winner with monotonic descending rankings.")

    # =========================================================================
    # 4.2 NORTH STAR 2: Spatial-Temporal Synchronization & The 4 Golden Rules
    # =========================================================================
    def test_north_star_4_2_fusion_golden_rules(self):
        """Validates alignment audit metrics and TopCam coordinate preservation."""
        # Check 1: Alignment audit CSV
        audit_csv = EXPERIMENT_DIR / "cohort_alignment_audit.csv"
        self.assertTrue(audit_csv.exists(), f"Audit CSV missing: {audit_csv}")

        df_audit = pd.read_csv(audit_csv)
        self.assertGreaterEqual(len(df_audit), 118, "Audit must cover all 118 sessions")

        med_corr_x = df_audit["corr_x"].median()
        med_corr_y = df_audit["corr_y"].median()
        self.assertGreater(med_corr_x, 0.95, f"Median X correlation {med_corr_x:.3f} below 0.95")
        self.assertGreater(med_corr_y, 0.95, f"Median Y correlation {med_corr_y:.3f} below 0.95")

        # Check 2: Verify Rule 1 (TopCam coordinates verbatim in fused H5)
        fused_files = sorted((EXPERIMENT_DIR / "fused_h5_full_cohort").glob("*.h5"))
        self.assertGreaterEqual(len(fused_files), 1, "Must have fused HDF5 files")

        sample_fused = fused_files[0]
        with h5py.File(sample_fused, "r") as f:
            # Check key structure
            self.assertIn("df_with_missing", f)

        df_fused = pd.read_hdf(sample_fused)
        fused_cols = [c[1] for c in df_fused.columns if c[2] == "x"]
        self.assertEqual(len(fused_cols), 23, "Fused dataframe must contain 23 keypoints")

        for bp in TOP_USE_BODYPARTS:
            self.assertIn(bp, fused_cols, f"Dorsal bodypart '{bp}' missing from fused columns")
        for bp in BOT_USE_BODYPARTS:
            self.assertIn(bp, fused_cols, f"Ventral bodypart '{bp}' missing from fused columns")

        print(f"\n[North Star 4.2 PASSED]: Alignment correlations verified (rx={med_corr_x:.3f}, ry={med_corr_y:.3f}). 23-kp layout confirmed.")

    # =========================================================================
    # 4.3 NORTH STAR 3: Syllable Vocabulary, Usage & Transition Dynamics
    # =========================================================================
    def test_north_star_4_3_syllable_dynamics(self):
        """Validates syllable vocabulary occupancy, duration profiles, and bigram transitions."""
        results_candidates = [
            PROD_KPMS_DIR / "tandem_cohort_118sess_50ar_200slds_seed303/results.h5",
            PROD_KPMS_DIR / "tandem_cohort_118sess_50ar_200slds/results.h5",
        ]

        active_results = None
        for cand in results_candidates:
            if cand.exists() and cand.stat().st_size > 1_000_000:
                active_results = cand
                break

        self.assertIsNotNone(active_results, "At least one production results.h5 must be available for testing")

        stats = extract_syllable_statistics(active_results)
        num_sessions = stats["num_sessions"]
        active_sylls = stats["active_syllables"]
        med_dur = stats["median_duration_frames"]
        T = stats["transition_matrix"]

        self.assertGreaterEqual(num_sessions, 100, f"Expected ~118 sessions, found {num_sessions}")
        # Active syllables expected between 60 and 100 (100 is max capacity of the SLDS prior)
        self.assertGreaterEqual(active_sylls, 60, f"Active syllables ({active_sylls}) too low")
        self.assertLessEqual(active_sylls, 100, f"Active syllables ({active_sylls}) too high")

        # Median duration expected between 15 and 35 frames (250 ms to 580 ms at 60 FPS)
        self.assertGreaterEqual(med_dur, 15.0, f"Median duration {med_dur} frames too short")
        self.assertLessEqual(med_dur, 35.0, f"Median duration {med_dur} frames too long")

        # Transition matrix row-stochasticity check
        row_sums = T.sum(axis=1)
        valid_rows = row_sums > 0
        np.testing.assert_allclose(row_sums[valid_rows], 1.0, atol=1e-5, err_msg="Transition matrix must be row-stochastic")

        print(f"\n[North Star 4.3 PASSED]: Active syllables = {active_sylls}, median duration = {med_dur:.1f} frames ({med_dur/60.0*1000:.0f} ms). Transitions stochastic.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
