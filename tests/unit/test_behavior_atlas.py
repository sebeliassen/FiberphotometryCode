"""Independent expectations for gap boundaries, alignment and factorization."""

import numpy as np
import pandas as pd
import pytest

from fpcode.adapters.behavior_sessions import verify_result_projection
from fpcode.behavior.phenotyping import analyze_frames, frame_kinematics
from fpcode.behavior.sequences.chains import edit_alignment, exact_occurrences, group_occurrences
from fpcode.behavior.sequences.prediction import HistoryPredictor
from fpcode.behavior.sequences.seqnmf import SeqNMF, reconstruct
from fpcode.behavior.variants import assign_execution_variants, fit_execution_variants
from fpcode.statistics.behavior_comparisons import adjust_fdr, compare_feature_atlas, independent_contrast
from fpcode.tracking.pose_measurements import body_relative_features
from fpcode.workflows.sequence_atlas import one_hot_segments


def test_dlc_pose_alignment_uses_native_index_not_row_position():
    from fpcode.adapters.behavior_sessions import pose_from_dlc_table

    columns = pd.MultiIndex.from_product([['tracker'], ['nose', 'tail_base'], ['x', 'y', 'likelihood']])
    table = pd.DataFrame([[2., 0., .9, 0., 0., .8], [4., 0., .9, 2., 0., .8]],
                         index=[101, 105], columns=columns)
    pose = pose_from_dlc_table(table, frames=np.array([100, 101, 105]),
                               bodyparts=('nose', 'tail_base'), clock='video',
                               unit='px', coordinate_frame='cropped_topcam')
    assert pose.time.values.tolist() == [100, 101, 105]
    assert not pose.valid[0].any()
    np.testing.assert_allclose(pose.coordinates[1:, 0, 0], [2., 4.])
    with pytest.raises(ValueError, match='unique.*increasing'):
        pose_from_dlc_table(pd.concat([table, table]), frames=np.array([101]),
                            bodyparts=('nose',), clock='video', unit='px', coordinate_frame='topcam')
    shifted = table.copy()
    shifted.index = [102, 106]
    moved = pose_from_dlc_table(shifted, frames=np.array([100, 101, 105]),
                                bodyparts=('nose',), clock='video', unit='px', coordinate_frame='topcam')
    assert not moved.valid.any()
    sentinel = table.copy()
    sentinel.loc[101, ('tracker', 'nose', 'likelihood')] = -1.
    masked = pose_from_dlc_table(sentinel, frames=np.array([101, 105]),
        bodyparts=('nose',), clock='video', unit='px', coordinate_frame='topcam', invalid_confidence_policy='mask')
    assert masked.valid[:, 0].tolist() == [False, True]
    assert sentinel.loc[101, ('tracker', 'nose', 'likelihood')] == -1.
    with pytest.raises(ValueError, match='confidence'):
        pose_from_dlc_table(sentinel, frames=np.array([101]),
            bodyparts=('nose',), clock='video', unit='px', coordinate_frame='topcam')


def test_animal_discovery_split_rejects_leakage_and_group_changes():
    from fpcode.workflows.sequence_atlas import validate_animal_splits

    sessions = pd.DataFrame({'session_id': ['a_pre', 'a_post', 'b', 'c'],
                             'animal_id': ['a', 'a', 'b', 'c'], 'group': ['PFF', 'PFF', 'Mono', 'Mono']})
    splits = {'train': ['a'], 'validation': ['b'], 'test': ['c']}
    assert validate_animal_splits(sessions, splits) == {'a': 'train', 'b': 'validation', 'c': 'test'}
    with pytest.raises(ValueError, match='overlap'):
        validate_animal_splits(sessions, {**splits, 'test': ['a', 'c']})
    with pytest.raises(ValueError, match='cover'):
        validate_animal_splits(sessions, {**splits, 'test': []})
    changed = sessions.copy()
    changed.loc[1, 'group'] = 'Mono'
    with pytest.raises(ValueError, match='group'):
        validate_animal_splits(changed, splits)


def test_native_projection_rejects_shift_and_overlap():
    arrays = {"frames": np.arange(7), "labels": np.array([-1, 2, 3, -1, 4, 4, -1])}
    results = {"a": {"syllable": np.array([2, 3])}, "b": {"syllable": np.array([4, 4])}}
    assert verify_result_projection(arrays, results, {"a": np.array([1, 2]), "b": np.array([4, 5])})["verified_labels"] == 4
    with pytest.raises(ValueError, match="differs"):
        verify_result_projection(arrays, results, {"a": np.array([0, 1]), "b": np.array([4, 5])})
    with pytest.raises(ValueError, match="overlapping"):
        verify_result_projection(arrays, results, {"a": np.array([1, 2]), "b": np.array([2, 3])})


def test_bouts_and_derivatives_preserve_gap_and_units():
    arrays = {"frames": np.arange(9), "labels": np.array([0, 0, 1, 1, -1, 0, 0, 1, 1]),
              "centroid": np.column_stack((np.arange(9) * 2., np.zeros(9))),
              "heading": np.zeros(9), "latent": np.zeros((9, 1)), "fps": np.array(10.)}
    result = analyze_frames(arrays, num_states=2, pause_speed_px_s=5, filter_frames=1)
    assert result["bouts"].duration_s.tolist() == [.2, .2, .2, .2]
    assert result["transitions"]["count"].tolist() == [2]
    assert result["summary"]["valid_duration_s"] == .8
    assert result["summary"]["mean_speed_px_s"] == 20.
    kin = frame_kinematics(arrays["centroid"], arrays["heading"], arrays["latent"],
                           arrays["frames"], arrays["labels"], 10., filter_frames=1)
    assert np.isnan(kin["speed_px_s"][4:6]).all()


def test_overlap_suppression_and_edit_operations():
    bouts = pd.DataFrame({"syllable_id": [0, 1, 0, 1, 0], "segment": [1] * 5})
    assert exact_occurrences(bouts, [0, 1, 0]) == [(0, 3)]
    assert edit_alignment([0, 1, 2], [0, 3, 1, 2]) == {
        "edit_distance": 1, "insertions": 1, "omissions": 0, "substitutions": 0}
    assert edit_alignment([0, 1, 2], [0, 2])["omissions"] == 1
    bouts.segment = [1, 1, 2, 2, 2]
    assert exact_occurrences(bouts, [0, 1, 0]) == [(2, 5)]


def test_occurrence_grouping_respects_competitors_and_segments():
    events = pd.DataFrame({"chain_id": [0, 0, 1, 0, 0], "segment": [1, 1, 1, 1, 2],
                           "start_s": [0., 2., 4., 6., 8.], "end_s": [1., 3., 5., 7., 9.]})
    result = group_occurrences(events, gap_seconds=5)
    assert result.repeat_position.tolist() == [1, 2, 1, 1, 1]


def test_zero_shift_reconstruction_has_no_wrap():
    w = np.array([[[1., 2.]]])
    h = np.array([[0., 0., 3.]])
    assert np.array_equal(reconstruct(w, h), [[0., 0., 3.]])


def test_seqnmf_recovers_repeating_known_pattern_and_masks_starts():
    x = np.zeros((3, 60), dtype=np.float32)
    for t in range(0, 60, 6):
        x[0, t], x[1, t + 1], x[2, t + 2] = 1., 1., 1.
    valid = np.zeros(60, dtype=bool)
    valid[::6] = True
    model = SeqNMF(1, 3, penalty=0, iterations=70, seed=7).fit(x, valid_starts=valid)
    assert np.linalg.norm(x - reconstruct(model.w, model.h)) / np.linalg.norm(x) < .01
    assert np.argmax(model.w[:, 0, :], axis=0).tolist() == [0, 1, 2]
    assert not model.h[:, ~valid].any()


def test_fdr_and_missing_support():
    assert np.allclose(adjust_fdr([.01, .04, .03, np.nan])[:3], [.03, .04, .04])
    assert np.isnan(adjust_fdr([.01, np.nan])[1])
    assert independent_contrast([1, 2], [5, 6])["status"] == "insufficient_animals"
    constant = independent_contrast([1., 1., 1. + 1e-15], [1., 1., 1.])
    assert constant["status"] == "constant"
    assert np.isnan(constant["hedges_g"])


def test_inference_does_not_count_repeated_sessions_as_animals():
    rows = [{"session_id": f"s{animal}_{repeat}", "animal_id": str(animal),
             "group": "first" if animal < 3 else "second", "condition": "pre", "family": "test",
             "feature": "value", "value": animal} for animal in range(6) for repeat in range(10)]
    output = compare_feature_atlas(pd.DataFrame(rows), groups=("first", "second"), untreated=("pre",),
                                  drug_condition="post", baseline_condition="pre", resamples=99)
    result = output["group_contrasts"].query("view == 'pre'").iloc[0]
    assert result.n_first == result.n_second == 3
    assert result.difference == 3


def test_execution_variants_separate_independent_physical_features():
    rng = np.random.default_rng(3)
    first = rng.normal([.2, 5., .2], [.01, .1, .01], size=(30, 3))
    second = rng.normal([.4, 10., .8], [.01, .1, .01], size=(30, 3))
    model = fit_execution_variants(np.r_[first, second])
    labels = assign_execution_variants(np.r_[first, second], model)
    assert len(np.unique(labels[:30])) == len(np.unique(labels[30:])) == 1
    assert labels[0] != labels[-1]
    assert model["seed_assignment_agreement"] == 1


def test_history_model_uses_matched_prediction_positions():
    predictor = HistoryPredictor(3, max_history=2).fit([np.array([0, 1, 2] * 20)])
    seq = [np.array([0, 1, 2] * 10)]
    zero, one = predictor.score(seq, 0), predictor.score(seq, 1)
    assert zero["n_predictions"] == one["n_predictions"] == 28
    assert one["cross_entropy_bits"] < zero["cross_entropy_bits"]


def test_motif_windows_never_join_recording_gaps():
    b = pd.DataFrame({"syllable_id": [0, 1, 2, 0, 1, 2], "segment": [1, 1, 1, 2, 2, 2]})
    x, mask, original = one_hot_segments(b, 3, 3)
    assert np.flatnonzero(mask).tolist() == [0, 6]
    assert original.tolist() == [0, 1, 2, -1, -1, -1, 3, 4, 5, -1, -1, -1]
    assert not x[:, 3:6].any()


def test_body_relative_paws_remove_translation_and_preserve_confidence_gaps():
    landmark = {name: i for i, name in enumerate(("nose", "tail", "neck", "left_shoulder", "right_shoulder",
                                                "left_hip", "right_hip", "left_front", "right_front", "left_hind", "right_hind"))}
    frame = np.array([[10., 0], [0, 0], [8, 0], [7, 2], [7, -2], [3, 2], [3, -2],
                      [6, 3], [6, -3], [2, 3], [2, -3]])
    pose = frame[None, :, :] + np.arange(4)[:, None, None]
    confidence = np.ones((4, 11))
    confidence[2, 7] = 0
    measured = body_relative_features(pose, confidence, landmarks=landmark, confidence_threshold=.5,
                                     frames=np.arange(4), fps=60)
    assert np.allclose(measured["body_length_px"], 10)
    assert np.allclose(measured["shoulder_width_ratio"], .4)
    assert measured["left_front_speed_bodylengths_s"][1] == 0
    assert np.isnan(measured["left_front_speed_bodylengths_s"][2:]).all()


def test_prospective_sensitivity_is_alpha_under_null_and_increases_with_n():
    from fpcode.statistics.followup_planning import two_group_power

    assert abs(two_group_power(0., 12, alpha=.05) - .05) < 1e-10
    assert two_group_power(.5, 40) > two_group_power(.5, 12)
    assert two_group_power(.5, 40, alpha=.005) < two_group_power(.5, 40)


def test_followup_ranking_flags_one_mouse_driven_direction():
    from fpcode.statistics.followup_planning import exploratory_effect_stability

    rows = []
    for group, values in (("first", [0.] * 6), ("second", [-1.] * 5 + [30.])):
        for animal, value in enumerate(values):
            for wave in ("early", "late"):
                rows.append({"animal_id": f"{group}{animal}", "group": group, "condition": wave,
                             "family": "usage", "feature": "state", "value": value})
    contrast = pd.DataFrame([{"family": "usage", "feature": "state", "view": "untreated_pooled",
                              "n_first": 6, "n_second": 6, "difference": 25 / 6, "hedges_g": .5}])
    result = exploratory_effect_stability(pd.DataFrame(rows), contrast, groups=("first", "second"),
                                          untreated=("early", "late"), consistency_waves=("early", "late")).iloc[0]
    assert result.loo_difference_min == -1
    assert result.leave_one_animal_out_direction_fraction < 1
    assert not result.meets_exploratory_heuristic




def test_analysis_interval_mask_preserves_native_clock_and_breaks_chain_support():
    from fpcode.behavior.analysis.organisation import exclusive_timeline, valid_frame_intervals
    from fpcode.workflows.behavior_atlas import mask_frame_cache_intervals

    source = {'frames': np.array([100, 101, 102, 103, 104, 107]),
              'labels': np.array([0, 0, 1, 1, 0, 0], dtype=np.int32),
              'centroid': np.ones((6, 2)), 'heading': np.ones(6),
              'latent': np.ones((6, 3)), 'fps': np.array(60.)}
    masked = mask_frame_cache_intervals(source, [[102, 104]])
    assert masked['labels'].tolist() == [0, 0, -1, -1, 0, 0]
    np.testing.assert_array_equal(masked['frames'], source['frames'])
    assert np.isnan(masked['centroid'][2:4]).all()
    assert np.isfinite(masked['centroid'][[0, 1, 4, 5]]).all()
    assert source['labels'].tolist() == [0, 0, 1, 1, 0, 0]
    assert np.isfinite(source['latent']).all()
    blocks = valid_frame_intervals(masked['frames'], masked['labels'])
    assert blocks.tolist() == [[100, 102], [104, 105], [107, 108]]
    calls = pd.DataFrame([{'chain_id': 'crossing', 'start_frame': 101,
                           'end_frame_exclusive': 105, 'selection_score': 1.}])
    selected, audit, timeline = exclusive_timeline(calls, blocks, fps=60., recording_end_frame=108)
    assert selected.empty
    assert audit.rejection_reason.tolist() == ['outside_valid_block']
    with pytest.raises(ValueError, match='half-open'):
        mask_frame_cache_intervals(source, [[102.5, 104.]])
