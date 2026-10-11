#!/usr/bin/env python3
"""
test_v2_visualization_repairs.py
================================
Regression test suite asserting the repaired behaviors for the V2 density
sampling and visualization pipeline.

Transforms the Codex reproduction findings (which demonstrated the original
bugs) into regression assertions guaranteeing:
1. Canonical half-open bout bounds (no off-by-one errors).
2. FPS-derived timing windows (40 frames at 60 FPS, 20 at 30 FPS).
3. Post-QC eligible tier routing.
4. Exact KPMS density-ratio exemplar selection with deterministic tie-breaking.
5. Elimination of silent rendering substitutions & duplicates (flagged blanks only).
6. Four-paw tracking QC (nose/tail cannot certify missing paws).
7. Robust BotCam pose fallback without arbitrary zero heading.
8. Circular heading median filtering on continuous segments.
9. Provenance sidecars & hash-based cache invalidation.
10. Grid layout dynamic derivation from G.
"""

import contextlib
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:
    class PytestShim:
        @staticmethod
        @contextlib.contextmanager
        def raises(exc_cls):
            try:
                yield
            except exc_cls:
                return
            except Exception as e:
                raise AssertionError(f"Expected {exc_cls}, but got {type(e)}") from e
            raise AssertionError(f"Expected {exc_cls}, but no exception was raised")

        @staticmethod
        def main(args):
            unittest.main()
    pytest = PytestShim()

from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "studies/multiview_experiments"
SCRIPT_DIR = EXP_DIR / "04_reports_and_figures"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fpcode.viz.v2 as vc  # noqa: E402


# =============================================================================
# 1. Canonical Half-Open Bout Bounds
# =============================================================================
def test_half_open_bouts_basic():
    # Sequence of 10 zeros, 40 ones, 10 twos
    seq = np.r_[np.zeros(10, dtype=int), np.ones(40, dtype=int), np.full(10, 2, dtype=int)]
    starts, ends = vc.half_open_bouts(seq)

    assert len(starts) == 3
    assert len(ends) == 3
    # True 40-frame bout must be reported as 40 frames: [10, 50)
    assert starts[1] == 10
    assert ends[1] == 50
    assert (ends[1] - starts[1]) == 40

    # Boundary bouts
    assert starts[0] == 0 and ends[0] == 10 and (ends[0] - starts[0]) == 10
    assert starts[2] == 50 and ends[2] == 60 and (ends[2] - starts[2]) == 10


def test_half_open_bouts_edge_cases():
    # Empty sequence
    starts, ends = vc.half_open_bouts([])
    assert len(starts) == 0 and len(ends) == 0

    # Single constant sequence
    starts, ends = vc.half_open_bouts(np.full(25, 7, dtype=int))
    assert len(starts) == 1
    assert starts[0] == 0
    assert ends[0] == 25
    assert (ends[0] - starts[0]) == 25

    # Alternating single-frame sequence
    alt = np.array([0, 1, 0, 1, 0])
    starts, ends = vc.half_open_bouts(alt)
    assert len(starts) == 5
    assert np.all(ends - starts == 1)


# =============================================================================
# 2. Timing Windows Derived from FPS
# =============================================================================
def test_timing_windows_fps_derivation():
    # 60 FPS: 1/6s pre (10 frames) + 0.5s post (30 frames) = 40 frames total
    win60 = vc.make_window(60.0, vc.DEFAULT_SCORE_PRE_SECONDS, vc.DEFAULT_SCORE_POST_SECONDS)
    assert win60["fps"] == 60.0
    assert win60["pre_frames"] == 10
    assert win60["post_frames"] == 30
    assert win60["total_frames"] == 40

    # 30 FPS: same seconds yield 5 pre + 15 post = 20 frames total
    win30 = vc.make_window(30.0, vc.DEFAULT_SCORE_PRE_SECONDS, vc.DEFAULT_SCORE_POST_SECONDS)
    assert win30["fps"] == 30.0
    assert win30["pre_frames"] == 5
    assert win30["post_frames"] == 15
    assert win30["total_frames"] == 20

    # Invalid FPS must raise ValueError
    with pytest.raises(ValueError):
        vc.make_window(0.0, 0.1, 0.5)


# =============================================================================
# 3. Post-QC Eligible Tier Routing
# =============================================================================
def test_choose_tier_from_eligible_count():
    G = 12
    assert vc.choose_tier(0, G) == "no_eligible_bouts"
    assert vc.choose_tier(5, G) == "tier_3_all_bouts"
    assert vc.choose_tier(12, G) == "tier_3_all_bouts"
    assert vc.choose_tier(13, G) == "tier_2_compact_neighborhood"
    assert vc.choose_tier(50, G) == "tier_2_compact_neighborhood"
    assert vc.choose_tier(51, G) == "tier_1_density"
    assert vc.choose_tier(1000, G) == "tier_1_density"


# =============================================================================
# 4. Density Selection & Tie Breaking
# =============================================================================
def test_tier1_density_exemplar_matching():
    rng = np.random.default_rng(42)
    # Background PCA
    bg = rng.normal(size=(1000, 4))
    global_nbrs = NearestNeighbors(n_neighbors=50).fit(bg)

    # State instances
    state = rng.normal(loc=0.5, scale=0.3, size=(200, 4))
    stable_keys = [(f"seg_{i:03d}", i * 10) for i in range(len(state))]
    ranks = vc.stable_ranks(stable_keys)

    result = vc.select_tier1_density(state, global_nbrs, ranks, G=12, n_neighbors=50)

    # Official KPMS calculation
    nbrs = NearestNeighbors(n_neighbors=50).fit(state)
    d, i = nbrs.kneighbors(state)
    dg, _ = global_nbrs.kneighbors(state)
    expected_ratio = (1.0 / d.mean(axis=1)) / (1.0 / dg.mean(axis=1))
    expected_exemplar = int(np.argmax(expected_ratio))

    assert result["anchor"] == expected_exemplar
    assert result["chosen"][0] == expected_exemplar
    assert len(result["chosen"]) == 12
    # Verify no duplicates
    assert len(set(result["chosen"])) == 12


def test_identical_feature_tie_guarantees_exemplar_at_zero():
    # 60 identical feature vectors
    state = np.zeros((60, 4))
    bg = np.zeros((100, 4))
    global_nbrs = NearestNeighbors(n_neighbors=50).fit(bg)
    stable_keys = [(f"seg_{i:03d}", i * 10) for i in range(60)]
    ranks = vc.stable_ranks(stable_keys)

    result = vc.select_tier1_density(state, global_nbrs, ranks, G=12, n_neighbors=50)

    # Exemplar must be anchor and at position 0
    assert result["anchor"] in result["chosen"]
    assert result["chosen"][0] == result["anchor"]
    assert len(result["chosen"]) == 12
    assert len(set(result["chosen"])) == 12
    # Remaining candidates must be sorted deterministically by stable ID rank
    assert result["chosen"][1:] == sorted(result["chosen"][1:])


# =============================================================================
# 5. Elimination of Render-Time Substitutions & Duplicates
# =============================================================================
def test_resolve_display_bouts_no_substitution_and_flagged_blanks():
    manifest_bouts = [
        {"bout_id": "segA:100", "segment_name": "segA", "onset_segment_frame": 100, "is_anchor": True, "anchor_type": "density_exemplar"},
        {"bout_id": "segB:200", "segment_name": "segB", "onset_segment_frame": 200, "is_anchor": False},
        {"bout_id": "segC:300", "segment_name": "segC", "onset_segment_frame": 300, "is_anchor": False},
    ]

    # Target validator rejects segA, accepts segB and segC
    def validator(b):
        if b["bout_id"] == "segA:100":
            return False, "tracking_dropped"
        return True, "ok"

    log_messages = []
    display = vc.resolve_display_bouts(manifest_bouts, validator, log=log_messages.append)

    assert len(display) == 3
    # Tile 0 must be a flagged blank with original bout_id preserved
    assert display[0]["is_blank"] is True
    assert display[0]["flagged_invalid"] is True
    assert display[0]["bout_id"] == "segA:100"
    assert display[0]["invalid_reason"] == "tracking_dropped"
    assert display[0]["is_anchor"] is True  # Anchor flag stays with tile 0

    # Tile 1 must remain segB without duplication!
    assert display[1]["bout_id"] == "segB:200"
    assert display[1]["is_blank"] is False
    assert display[1]["is_anchor"] is False  # Never inherits anchor status

    # Tile 2 must remain segC
    assert display[2]["bout_id"] == "segC:300"
    assert display[2]["is_blank"] is False

    # Assert rendered equals manifest
    vc.assert_rendered_equals_manifest(display, manifest_bouts)


# =============================================================================
# 6. Four-Paw Tracking QC
# =============================================================================
def test_four_paw_qc_rejection():
    cfg = vc.QCConfig(
        conf_threshold=0.5,
        min_dorsal_valid_fraction=0.8,
        min_paw_valid_fraction=0.8,
        max_paw_gap_seconds=0.1
    )
    fps = 60.0
    W = 40  # 40-frame window

    # Perfect dorsal keypoints (indices 0..16)
    valid_window = np.ones((W, 21), dtype=bool)

    # All paws (indices 17..20) are untracked / low-confidence
    valid_window[:, vc.PAW_IDXS] = False

    summary = vc.summarize_window_qc(valid_window)
    passed, reasons = vc.qc_decision(summary, cfg, fps)

    assert passed is False
    assert "qc_paw_fraction" in reasons
    assert summary["min_dorsal_valid_fraction"] == 1.0
    assert summary["min_paw_valid_fraction"] == 0.0
    assert vc.collapse_qc_reason(reasons) == "qc_paws"


# =============================================================================
# 7. BotCam Pose & Paw Fallback
# =============================================================================
def test_bot_pose_fallback_with_actual_paw_names():
    # DataFrame with valid paws but no nose/tail
    n = 100
    bp = ["nose_tip", "tail_base", "left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw"]
    cols = pd.MultiIndex.from_product([["dlc"], bp, ["x", "y", "likelihood"]])
    df = pd.DataFrame(np.nan, index=np.arange(n), columns=cols)

    # Nose and tail missing
    df[("dlc", "nose_tip", "likelihood")] = 0.0
    df[("dlc", "tail_base", "likelihood")] = 0.0

    # 4 paws tracked around x=300, y=400 with high confidence
    # Front paws at y=390, hind paws at y=410 -> vector points along -y (angle = -pi/2)
    for p, (px, py) in [
        ("left_front_paw", (290.0, 390.0)),
        ("right_front_paw", (310.0, 390.0)),
        ("left_hind_paw", (290.0, 410.0)),
        ("right_hind_paw", (310.0, 410.0)),
    ]:
        df[("dlc", p, "x")] = px
        df[("dlc", p, "y")] = py
        df[("dlc", p, "likelihood")] = 0.95

    # Tier 2: Paw longitudinal axis heading (-pi/2)
    c, h, is_valid = vc.compute_robust_bot_pose(df, "dlc", 50, search_window=5)

    assert is_valid is True
    assert np.isclose(c[0], 300.0, atol=1e-3)
    assert np.isclose(c[1], 400.0, atol=1e-3)
    assert np.isclose(h, -np.pi / 2.0, atol=1e-3)

    # Tier 3: TopCam heading fallback when paws cannot establish orientation
    # Single paw only (cannot establish paw axis)
    df_single = df.copy()
    for p in ["right_front_paw", "left_hind_paw", "right_hind_paw"]:
        df_single[("dlc", p, "likelihood")] = 0.1
    c2, h2, is_valid2 = vc.compute_robust_bot_pose(
        df_single, "dlc", 50, search_window=5, top_heading_fallback=0.0
    )
    # top_heading=0.0 in TopCam maps to atan2(sin(0), -cos(0)) = atan2(0, -1) = pi in BotCam
    assert np.isclose(h2, np.pi, atol=1e-3)

    # Tier 4: Total failure returns NaN
    c3, h3, is_valid3 = vc.compute_robust_bot_pose(
        df_single, "dlc", 50, search_window=5, top_heading_fallback=None
    )
    assert np.isnan(h3)


# =============================================================================
# 8. Circular Heading Median Filtering
# =============================================================================
def test_circular_heading_filter_removes_glitch():
    n = 101
    centroids = np.zeros((n, 2))
    headings = np.zeros(n)
    # Isolated 90-degree glitch at frame 50
    headings[50] = np.pi / 2.0

    c_filt, h_filt = vc.filter_segment_pose(centroids, headings, filter_size=9)

    # Filtered heading at glitch frame must be close to 0 (glitch removed)
    assert np.abs(h_filt[50]) < 1e-4
    assert np.allclose(h_filt, 0.0, atol=1e-4)


# =============================================================================
# 9. Provenance & Cache Decisions
# =============================================================================
def test_cache_decision_invalidates_on_key_change(tmp_path=None):
    if tmp_path is None:
        with tempfile.TemporaryDirectory() as td:
            _run_cache_decision_test(Path(td))
    else:
        _run_cache_decision_test(tmp_path)


def _run_cache_decision_test(tmp_path: Path):
    out_file = tmp_path / "movie.mp4"
    out_file.write_bytes(b"dummy_video_bytes_12345")
    sidecar = tmp_path / "movie.mp4.provenance.json"

    key_orig = {"manifest_sha": "abc", "code_sha": "123", "syllable": 1}
    vc.write_json_atomic(sidecar, {
        "cache_key": key_orig,
        "outputs": {"movie.mp4": {"size": out_file.stat().st_size}}
    })

    # Same key -> cache hit
    reuse, reason = vc.cache_decision([out_file], key_orig)
    assert reuse is True
    assert "matches" in reason

    # Changed code sha -> cache miss!
    key_new = copy.deepcopy(key_orig)
    key_new["code_sha"] = "456"
    reuse, reason = vc.cache_decision([out_file], key_new)
    assert reuse is False
    assert "cache key changed" in reason


# =============================================================================
# 10. Dynamic Grid Layout
# =============================================================================
def test_grid_layout():
    assert vc.grid_layout(12) == (3, 4)
    assert vc.grid_layout(24) == (4, 6)
    # Non-standard grid sizes
    rows, cols = vc.grid_layout(16)
    assert rows * cols >= 16


class V2VisualizationRepairs(unittest.TestCase):
    # pytest collects the functions above. Keep this adapter for direct unittest
    # execution without counting and running the same twelve cases twice.
    __test__ = False

    def test_01_half_open_bouts_basic(self):
        test_half_open_bouts_basic()

    def test_02_half_open_bouts_edge_cases(self):
        test_half_open_bouts_edge_cases()

    def test_03_timing_windows_fps_derivation(self):
        test_timing_windows_fps_derivation()

    def test_04_choose_tier_from_eligible_count(self):
        test_choose_tier_from_eligible_count()

    def test_05_tier1_density_exemplar_matching(self):
        test_tier1_density_exemplar_matching()

    def test_06_identical_feature_tie_guarantees_exemplar_at_zero(self):
        test_identical_feature_tie_guarantees_exemplar_at_zero()

    def test_07_resolve_display_bouts_no_substitution(self):
        test_resolve_display_bouts_no_substitution_and_flagged_blanks()

    def test_08_four_paw_qc_rejection(self):
        test_four_paw_qc_rejection()

    def test_09_bot_pose_fallback_with_actual_paw_names(self):
        test_bot_pose_fallback_with_actual_paw_names()

    def test_10_circular_heading_filter_removes_glitch(self):
        test_circular_heading_filter_removes_glitch()

    def test_11_cache_decision_invalidates_on_key_change(self):
        test_cache_decision_invalidates_on_key_change()

    def test_12_grid_layout(self):
        test_grid_layout()


if __name__ == "__main__":
    unittest.main(verbosity=2)
