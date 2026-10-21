"""Independent resolution, gap and mixture expectations."""

import numpy as np
import pandas as pd
import pytest

from fpcode.behavior.bouts import run_boundaries
from fpcode.behavior.resolutions import member_composition, remap_states, standardised_member_execution


def test_mapping_merges_adjacent_bouts_but_never_missing_data():
    z = remap_states(np.array([0, 0, 1, 1, -1, 0, 1, 2]), np.array([0, 0, 1]))
    assert z.tolist() == [0, 0, 0, 0, -1, 0, 0, 1]
    starts, ends = run_boundaries(z, frame_indices=np.arange(8))
    assert starts.tolist() == [0, 4, 5, 7]
    assert ends.tolist() == [4, 5, 7, 8]
    starts, _ = run_boundaries(np.array([0, 0]), frame_indices=np.array([2, 4]))
    assert starts.tolist() == [0, 1]
    with pytest.raises(ValueError):
        remap_states(np.array([3]), np.array([0, 1]))


def test_fixed_mixture_removes_a_pure_composition_difference():
    rows = [{"session_id": sid, "syllable_id": state, "time_fraction": share, "speed": speed}
            for sid, shares in (("a", [.9, .1]), ("b", [.1, .9]))
            for state, (share, speed) in enumerate(zip(shares, [10., 100.]))]
    table = pd.DataFrame(rows)
    result = standardised_member_execution(table, np.array([0, 0]), np.array([.5, .5]),
                                           identity=["session_id"], features=["speed"])
    assert result.fixed_mixture_raw_execution.tolist() == [55., 55.]
    assert result.observed_mixture_raw_execution.tolist() == [19., 91.]
    table.loc[table.session_id == "a", "time_fraction"] = 0
    absent = member_composition(table, np.array([0, 0]), identity=["session_id"])
    assert absent.loc[absent.session_id == "a", "within_cluster_fraction"].isna().all()


def test_missing_reference_weight_does_not_fabricate_standardised_execution():
    table = pd.DataFrame({"session_id": ["a", "a"], "syllable_id": [0, 1],
                          "time_fraction": [.5, .5], "speed": [10., np.nan]})
    result = standardised_member_execution(table, np.array([0, 0]), np.array([.2, .8]),
                                           identity=["session_id"], features=["speed"])
    assert result.reference_weight_coverage.iloc[0] == .2
    assert np.isnan(result.fixed_mixture_raw_execution.iloc[0])


def test_karen_policy_preserves_direction_until_broad_level_and_splits_unresolved():
    from studies.karen_pff_vs_monomer.analysis.karen_hierarchy import build_hierarchy

    annotation = pd.DataFrame({"syllable_id": np.arange(100), "description": [f"state{i}" for i in range(100)],
                               "category": ["Unassigned"] * 100})
    mapping, _, members = build_hierarchy(annotation)
    assert {key: value.max() + 1 for key, value in mapping.items()} == {
        "L1": 100, "L2": 93, "L3": 22, "L4": 14, "L3_unresolved_separate": 69, "L4_unresolved_separate": 61}
    assert mapping["L2"][3] == mapping["L2"][4]
    assert mapping["L3"][8] == mapping["L3"][71] != mapping["L3"][20]
    assert mapping["L4"][8] == mapping["L4"][20]
    assert mapping["L3"][61] == mapping["L3"][62]
    assert mapping["L3_unresolved_separate"][61] != mapping["L3_unresolved_separate"][62]
    assert mapping["L3_unresolved_separate"][12] != mapping["L3_unresolved_separate"][18]
    assert len(members) == 600


def test_anchored_patterns_record_collapse_and_do_not_bridge_a_gap(tmp_path):
    from fpcode.workflows.anchored_sequences import measure_anchored_sequences

    bouts_dir = tmp_path / "measurements/bouts_by_session"
    bouts_dir.mkdir(parents=True)
    bouts = pd.DataFrame({"syllable_id": [0, 1, 0, 1, 0], "segment": [0, 0, 0, 1, 1],
                          "duration_s": [.1] * 5, "mean_speed_px_s": [10.] * 5,
                          "start_frame": [0, 6, 12, 30, 36], "end_frame_exclusive": [6, 12, 18, 36, 42]})
    bouts.to_csv(bouts_dir / "s.csv", index=False)
    sessions = pd.DataFrame({"session_id": ["s"], "animal_id": ["a"], "group": ["g"], "condition": ["wave"]})
    catalog = pd.DataFrame({"chain_id": ["collapsed", "kept"], "tokens": ["0 1 0", "0 2 0"]})
    measured = measure_anchored_sequences(sessions, tmp_path / "measurements", tmp_path / "anchors", catalog,
                                         np.array([0, 0, 1]))
    audit = pd.read_csv(tmp_path / "anchors/anchor_mapping_audit.csv")
    assert audit.retained.tolist() == [False, True]
    rate = measured[measured.feature == "anchor_kept/occurrences_per_min"].value.iloc[0]
    assert rate == 120.
    empty = measure_anchored_sequences(sessions, tmp_path / "measurements", tmp_path / "empty", catalog.iloc[:1],
                                      np.array([0, 0, 1]))
    assert empty.empty
