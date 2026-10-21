"""Actual formerly missing metrics APIs preserve tiny independent study results."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from studies.kinematics_ulrik.lib import label_mapping, metrics


def session(labels, **columns):
    return SimpleNamespace(all_data_df=pd.DataFrame({"syllable": labels, **columns}),
                           metadata={"animal_id": "A", "genotype": "WT", "drug": "Saline"})


def test_compatibility_usage_and_quantitative_robust_bout_filter_are_callable():
    data = session([0, 0, 1, 0, 2, 2])
    group = metrics.get_group_usage([data], min_instances=2)
    assert group.Syllable.tolist() == [0]
    assert group.Frequency.tolist() == [1.0]
    quantitative = metrics.get_quantitative_group_usage(
        [data], {0: "Left_Low", 1: "Left_Low", 2: "Right_High"}, min_instances=2)
    left = quantitative[quantitative.Category == "Left_Low"].iloc[0]
    assert left.Time_Pct == 0.5  # Retains the full six-frame denominator.
    assert left.Onset_Count == 1
    assert quantitative.Time_Pct.sum() == 0.5
    assert quantitative.Onset_Count.sum() == 1


def test_compatibility_manual_semantic_usage_and_missing_boundary_transitions():
    usage = metrics.get_manual_group_usage([session([0, 0, 1, 0, 2, 2])], {0: "walk", 1: "walk", 2: "rear"})
    usage = usage.set_index("Label")
    np.testing.assert_allclose(usage.loc[["walk", "rear"], "Time_Pct"], [2/3, 1/3])
    np.testing.assert_array_equal(usage.loc[["walk", "rear"], "Onset_Count"], [1, 1])
    matrix, individual = metrics.get_manual_group_transition_matrix(
        [session([0, 0, 1, 2, np.nan, 0, 2])], {0: "walk", 1: "walk", 2: "rear"},
        return_distributions=True)
    assert len(individual) == 1
    np.testing.assert_array_equal(matrix.loc[["walk", "rear"], ["walk", "rear"]], [[0, 1], [0, 0]])


def test_split_usage_and_transition_helpers_accept_empty_sessions():
    empty = session([])
    quantitative = metrics.get_quantitative_group_usage([empty], {0: "Left_Low"}, min_instances=2)
    assert len(quantitative) == 9
    assert quantitative.Time_Pct.eq(0).all()
    assert quantitative.Onset_Count.eq(0).all()
    matrix, individual = metrics.get_manual_group_transition_matrix([empty], {0: "walk"}, return_distributions=True)
    assert individual == []
    np.testing.assert_array_equal(matrix, [[0]])


def test_split_fingerprint_resolves_retained_instance_helper():
    data = session([0, 0, 1, 1, 0, 0], velocity_px_s=[3.] * 6, acceleration=[1.] * 6,
                   centroid_x=np.arange(6), centroid_y=[0.] * 6)
    result = label_mapping.get_fingerprint_data([data], min_instances=2)
    assert result.syllable.tolist() == [0]
    np.testing.assert_allclose(result.mean_duration, [2/30])
    np.testing.assert_allclose(result.mean_velocity, [3])
    assert result.Animal.tolist() == ["A"]


def test_compatibility_loaders_are_real_split_callables(tmp_path):
    manual = tmp_path / "manual.csv"
    manual.write_text("project,syllable,label\nAMPH,2,walk\nAMPH,3,\n")
    assert metrics.load_manual_labels(manual) == {"AMPH": {2: "walk"}}
    quantitative = tmp_path / "quantitative.csv"
    quantitative.write_text("syllable,Vigor,Turn\n2,High,Left\n")
    assert metrics.load_quantitative_map(quantitative) == {2: "Left_High"}
