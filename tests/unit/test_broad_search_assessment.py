"""Independent expectations for study-level physical diagnostics."""
import numpy as np

from studies.sequence_viz.assess_broad_search import call_profiles, physical_distances, training_context


def test_analytical_cross_animal_distance_matches_explicit_pairs():
    values = np.array([[0., 1.], [2., 3.], [1., 4.], [3., 2.],
                       [10., 5.], [12., 7.], [11., 8.], [13., 6.]])
    labels = np.array(["a"]*4+["b"]*4)
    subjects = np.array(["1", "1", "2", "2"]*2)
    result = physical_distances(values, labels, subjects, [0, 1])
    within, between = [], []
    for token in ("a", "b"):
        first = values[(labels == token) & (subjects == "1")]
        second = values[(labels == token) & (subjects == "2")]
        within.append(np.mean([(a-b)**2 for a in first for b in second]))
        other = "b" if token == "a" else "a"
        distances = []
        for subject in ("1", "2"):
            first = values[(labels == token) & (subjects == subject)]
            second = values[(labels == other) & (subjects != subject)]
            distances.append(np.mean([(a-b)**2 for a in first for b in second]))
        between.append(np.mean(distances))
    np.testing.assert_allclose(result["within"], np.mean(within), rtol=1e-13)
    np.testing.assert_allclose(result["between"], np.mean(between), rtol=1e-13)
    np.testing.assert_allclose(result["separation"], np.mean([1-a/b for a, b in zip(within, between)]), rtol=1e-13)


def test_same_animal_and_unused_empty_columns_are_unavailable():
    values = np.arange(8.)[:, None]
    labels = np.array(["a"]*4+["b"]*4)
    subjects = np.array(["1"]*8)
    assert physical_distances(values, labels, subjects, [0])["separation"] is None
    assert physical_distances(values, labels, subjects, [])["separation"] is None


def test_native_time_phase_weights_and_source_conditioning():
    record = dict(start_frames=np.array([0, 3]), stop_frames=np.array([3, 9]),
                  values=np.array([[0.], [6.]]), parent_ids=np.array(["a", "b"]),
                  subject="1", recording="r", fps=3.)
    context = dict(location=np.array([0.]), scale=np.array([1.]),
                   parent_expected={"a": np.array([0.]), "b": np.array([6.])})
    call = dict(start_frame=0, stop_frame=9, type_id="routine")
    result = call_profiles(record, [call], context)[0]
    np.testing.assert_array_equal(result["profile"], [0, 6, 6])
    np.testing.assert_array_equal(result["residual"], [0, 0, 0])
    assert result["duration_s"] == 3.
    # A phase can intersect a bout without inventing an event boundary.
    record["start_frames"][1] = 4
    record["stop_frames"][0] = 4
    result = call_profiles(record, [call], context)[0]
    np.testing.assert_allclose(result["profile"], [0, 4, 6])


def test_held_values_do_not_change_scales_or_source_expectations():
    train = dict(split="train", recording="train", feature_names=["duration_s"],
                 values=np.array([[0.], [2.], [4.], [6.]]), parent_ids=np.array(["a", "a", "b", "b"]))
    held = dict(split="held", recording="held", feature_names=["duration_s"],
                values=np.array([[1e9]]), parent_ids=np.array(["a"]))
    context = training_context([train, held])
    np.testing.assert_array_equal(context["location"], [3.])
    np.testing.assert_array_equal(context["scale"], [3.])
    np.testing.assert_allclose(context["parent_expected"]["a"], [-2/3])


def test_unseen_source_state_residual_stays_missing():
    record = dict(start_frames=np.array([0]), stop_frames=np.array([9]), values=np.array([[2.]]),
                  parent_ids=np.array(["unseen"]), subject="1", recording="r", fps=3.)
    context = dict(location=np.array([0.]), scale=np.array([1.]), parent_expected={})
    result = call_profiles(record, [dict(start_frame=0, stop_frame=9, type_id="r")], context)[0]
    assert np.isnan(result["residual"]).all()
