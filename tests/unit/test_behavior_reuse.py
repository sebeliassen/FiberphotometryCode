"""Independent tiny expectations for reusable behavior and real study adapters."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fpcode.behavior.bouts import run_boundaries
from fpcode.behavior.kinematics import detect_movement_initiations, movement_changepoint_score
from fpcode.behavior.transitions import adjacent_pairs, compute_transition_matrix
from studies.kinematics_ulrik.lib.aggregation import get_group_transition_matrix
from studies.kinematics_ulrik.lib.micro_kinematics import extract_bouts as study_bouts
from studies.kinematics_ulrik.lib.motifs import extract_ngrams, get_label_mapping
from studies.kinematics_ulrik.lib.networks import build_transition_matrix
from studies.replications_datta.datta.lib import snippets
from studies.replications_datta.datta.panel_1f import run_panel_1f_unified as unified


def session(labels, **columns):
    return SimpleNamespace(all_data_df=pd.DataFrame({"syllable": labels, **columns}), metadata={})


def test_run_bounds_split_original_frame_gaps_without_joining_missing_labels():
    labels = np.array([4, 4, -1, 4, 4, 4, 5])
    frames = np.array([100, 101, 102, 103, 110, 111, 112])
    starts, stops = run_boundaries(labels, frame_indices=frames)
    np.testing.assert_array_equal(starts, [0, 2, 3, 4, 6])
    np.testing.assert_array_equal(stops, [2, 3, 4, 6, 7])
    np.testing.assert_array_equal(adjacent_pairs(labels, collapse_runs=True, frame_indices=frames),
                                  [[4, -1], [-1, 4], [4, 5]])
    np.testing.assert_array_equal(adjacent_pairs(labels, frame_indices=frames),
                                  [[4, 4], [4, -1], [-1, 4], [4, 4], [4, 5]])
    np.testing.assert_array_equal(labels, [4, 4, -1, 4, 4, 4, 5])
    assert run_boundaries([])[0].size == 0
    assert adjacent_pairs([], collapse_runs=True).shape == (0, 2)


@pytest.mark.parametrize("frames", [[0, 0], [1, 0], [0.0, 1.0], [0]])
def test_original_frame_policy_rejects_ambiguous_frame_maps(frames):
    with pytest.raises(ValueError, match="frame indices"):
        run_boundaries([1, 1], frame_indices=frames)


def test_frame_counts_do_not_connect_recordings_or_missing_states():
    counts = compute_transition_matrix([[0, 0, 1], [2, -1, 0]], num_states=3, normalize=False)
    np.testing.assert_array_equal(counts, [[1, 1, 0], [0, 0, 0], [0, 0, 0]])


def test_group_weighting_is_per_session_and_network_weighting_is_pooled():
    sessions = [session([0, 1, 0, 1, 0, 2]), session([0, 2, 0])]
    grouped = get_group_transition_matrix(sessions, ordered_syllables=[0, 1, 2])
    np.testing.assert_allclose(grouped, [[0, 1/3, 2/3], [0.5, 0, 0], [0.5, 0, 0]])
    pooled, frequency = build_transition_matrix(sessions, top_n=3)
    np.testing.assert_allclose(pooled.reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0),
                               [[0, 0.5, 0.5], [1, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(frequency.reindex([0, 1, 2]), [4/7, 2/7, 1/7])


def test_study_missing_and_unselected_states_break_bigrams():
    grouped = get_group_transition_matrix([session([0, np.nan, 2, 7, 0])], ordered_syllables=[0, 2])
    np.testing.assert_array_equal(grouped, np.zeros((2, 2)))
    pooled, _ = build_transition_matrix([session([0, np.nan, 2, 0, 1, 0])], top_n=3)
    assert pooled.reindex(columns=[0, 1, 2], fill_value=0).loc[0, 2] == 0
    assert pooled.loc[2, 0] == 1
    assert get_group_transition_matrix([session([])], ordered_syllables=[0, 2]).shape == (2, 2)


def test_motifs_collapse_semantic_labels_before_counting_and_keep_gap_boundaries():
    data = session([0, 1, 2, np.nan, 0, 1, 2])
    motifs = extract_ngrams(data, n=2, mapping={0: "walk", 1: "walk", 2: "rear"})
    assert motifs.to_dict() == {"walk -> rear": 2}
    assert "rear -> walk" not in motifs.index


def test_manual_mapping_uses_actual_loader_and_preserves_standard_label_policy(tmp_path, monkeypatch):
    labels_dir = tmp_path / "syllable_generation/analysis/ulrik_kinematics"
    labels_dir.mkdir(parents=True)
    path = labels_dir / "manual_labels.csv"
    path.write_text("project,syllable,label\nAMPH,1,left turn\nAMPH,2,dash\nAMPH,3,rear\nAMPH,4,no video\n")
    monkeypatch.setattr("studies.kinematics_ulrik.lib.motifs.get_repo_root", lambda: tmp_path)
    assert get_label_mapping([], style="standard") == {1: "left turn", 2: "walk/dash", 3: "rear"}
    assert get_label_mapping([], style="raw", manual_labels_path=path) == {
        1: "left turn", 2: "dash", 3: "rear", 4: "no video"}


def test_micro_bouts_preserve_observed_span_duration_and_nan_summaries():
    data = session([0, 0, np.nan, 1, 1, 1, 2], frame_index=[0, 3, 4, 10, 11, 12, 13],
                   velocity_px_s=[1, 3, 99, np.nan, 6, 9, 100],
                   angular_velocity=[1, -3, 100, 40, -50, 5, 4])
    original = data.all_data_df.copy(deep=True)
    bouts = study_bouts(data)
    np.testing.assert_array_equal(bouts.Bout_ID, [1, 3])
    np.testing.assert_array_equal(bouts.Frame_Count, [2, 3])
    np.testing.assert_allclose(bouts.Duration, [3/30, 2/30])
    np.testing.assert_allclose(bouts.Mean_Velocity, [2, 7.5])
    np.testing.assert_allclose(bouts.Mean_Angular, [2, 85/3])
    np.testing.assert_allclose(bouts.Median_Angular, [2, 40])
    pd.testing.assert_frame_equal(data.all_data_df, original)


def test_micro_heading_rate_uses_metadata_while_duration_uses_native_30fps():
    data = session([1] * 6, frame_index=np.arange(6), heading=[-3, -3, 3, 3, 3, 3])
    data.metadata["fs"] = 10.0
    bouts = study_bouts(data)
    assert bouts.Duration.iloc[0] == pytest.approx(5/30)
    assert bouts.Mean_Angular.iloc[0] == pytest.approx((2*np.pi - 6)*10/6)
    data.metadata["fs"] = 20.0
    assert study_bouts(data).Mean_Angular.iloc[0] == pytest.approx((2*np.pi - 6)*20/6)


def test_movement_score_scaling_and_nan_policies_have_independent_expectations():
    velocity = np.array([0., 1., 2., 3., 2., 1., 0.])
    expected = np.array([0., 1., 1., 1., -1., -1., -1.]) * (7/np.sqrt(52))**3
    score = movement_changepoint_score(velocity, 2, smoothing_seconds=0)
    np.testing.assert_allclose(score, expected)
    np.testing.assert_allclose(movement_changepoint_score(velocity, 2, derivative_scale="per_second",
                                                        smoothing_seconds=0), expected * 8)
    missing = np.array([0., np.nan, 2., 0.])
    # Valid velocities have mean 2/3 and standard deviation sqrt(8)/3.
    filled_z = np.array([-2/np.sqrt(8), 0, 4/np.sqrt(8), -2/np.sqrt(8)])
    np.testing.assert_allclose(movement_changepoint_score(missing, 1, missing="zero", smoothing_seconds=0),
                               np.diff(filled_z, prepend=filled_z[0])**3)
    propagated = movement_changepoint_score(missing, 1, missing="propagate", smoothing_seconds=0)
    assert np.isnan(propagated[1:]).all()
    np.testing.assert_array_equal(missing, [0, np.nan, 2, 0])


def test_real_datta_detectors_preserve_distinct_derivative_policies():
    df = pd.DataFrame({"velocity_px_s": [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]})
    assert snippets.detect_movement_initiations(df, 2).size == 0
    np.testing.assert_array_equal(unified.detect_movement_initiations(df, 2), [3])
    for data in ([], [3, 3, 3], [np.nan, np.nan]):
        frame = pd.DataFrame({"velocity_px_s": data})
        assert snippets.detect_movement_initiations(frame, 80).size == 0
        assert unified.detect_movement_initiations(frame, 80).size == 0
    assert detect_movement_initiations([], 80).size == 0


def test_syllable_onset_adapters_keep_nullable_and_float_missingness_policies():
    nullable = pd.DataFrame({"labels": pd.Series([1, 1, pd.NA, 2, 3], dtype="Int64")})
    np.testing.assert_array_equal(snippets.detect_syllable_onsets(nullable), [4])
    np.testing.assert_array_equal(unified.detect_syllable_onsets(nullable.rename(columns={"labels": "syllable"})),
                                  [0, 3, 4])
