"""Independent numerical checks for the bounded execution-distance policy."""

import numpy as np
import pytest

from studies.karen_pff_vs_monomer.analysis.execution_similarity import (
    distinct_execution_neighbours,
    exemplar_distances,
    fit_scaling,
    kinematics,
    phase_blocks,
    regularized_whitener,
    scaled_blocks,
    weighted_vectors,
)


def test_overlapping_other_mouse_calls_do_not_cast_duplicate_neighbour_votes():
    distances = np.array([[0., 1., 2., 3.]])
    neighbours, valid = distinct_execution_neighbours(distances, ["a", "a", "b", "a"],
                                                      [0, 5, 0, 10], [10, 12, 10, 20], count=3)
    # a:[5,12) overlaps the first candidate; a:[10,20) is half-open adjacent.
    np.testing.assert_array_equal(neighbours, [[0, 2, 3]])
    assert valid.all()


def test_regularized_whitening_has_correct_diagonal_and_correlated_units():
    np.testing.assert_allclose(regularized_whitener(np.diag([4., 9.]), .1), np.diag([.5, 1 / 3]))
    cov = np.array([[2., 1.], [1., 2.]])
    w = regularized_whitener(cov, .1)
    np.testing.assert_allclose(w @ np.array([[2., .9], [.9, 2.]]) @ w.T, np.eye(2), atol=1e-12)


def test_native_motion_has_physical_units_signed_turns_and_no_gap_jump():
    frames = np.r_[np.arange(60), np.arange(120, 180)]
    xy = np.column_stack([frames.astype(float), np.zeros(120)])
    xy[60:, 0] += 1000  # disconnected support must not become a speed spike
    heading = np.r_[np.arange(60) / 60, -np.arange(60) / 60]
    v = kinematics(frames, xy, heading)
    np.testing.assert_allclose(v[10:50, 0], np.log1p(60), atol=1e-12)
    np.testing.assert_allclose(v[70:110, 0], np.log1p(60), atol=1e-12)
    np.testing.assert_allclose(v[10:50, 1], np.arcsinh(1), atol=1e-12)
    np.testing.assert_allclose(v[70:110, 1], -np.arcsinh(1), atol=1e-12)
    np.testing.assert_allclose(v, kinematics(frames, xy, heading + 2 * np.pi), atol=1e-12)


def test_block_weight_is_not_multiplied_by_feature_count():
    blocks = {"posture": np.ones((1, 90)), "movement": np.ones((1, 18)), "tempo": np.ones((1, 3))}
    x = weighted_vectors(blocks, dict(posture=.45, movement=.45, tempo=.10))
    assert np.sum(x[0, :90] ** 2) == pytest.approx(.45)
    assert np.sum(x[0, 90:108] ** 2) == pytest.approx(.45)
    assert np.sum(x[0, 108:] ** 2) == pytest.approx(.10)


def test_unknown_semantics_never_make_two_executions_identical():
    blocks = {"posture": np.array([[0.], [1.], [2.]]), "movement": np.array([[0.], [1.], [2.]]),
              "tempo": np.array([[0.], [1.], [2.]]), "semantic": np.zeros((3, 9), int)}
    d, e, parts = exemplar_distances(blocks, ["a", "b", "c"], dict(posture=.45, movement=.35, tempo=.10, semantic=.10))
    assert e == 1
    np.testing.assert_allclose(d, [1., 0., 1.])
    assert np.isnan(parts["semantic"]).all()
    np.testing.assert_array_equal(parts["semantic_shared_fraction"], 0)


def test_one_known_landmark_has_one_ninth_of_full_semantic_weight():
    physical = np.array([[0.], [2.], [4.]])
    labels = np.ones((3, 9), int)
    labels[2] = 0
    labels[2, 0] = 2
    b = dict(posture=physical, movement=physical, tempo=physical, semantic=labels)
    d, exemplar, parts = exemplar_distances(b, ["a", "b", "c"], dict(posture=.45, movement=.35, tempo=.1, semantic=.1))
    assert exemplar == 1
    assert parts["semantic_shared_fraction"][2] == pytest.approx(1 / 9)
    assert d[2] == pytest.approx((.9 * 4 + .1 / 9) / (.9 + .1 / 9))


def test_phase_representation_keeps_bout_order_and_real_durations():
    track = dict(frames=np.arange(60), labels=np.r_[np.zeros(10), np.ones(20), np.full(30, 2)].astype(int),
                 latent=np.tile(np.arange(60)[:, None], (1, 10)))
    blocks = phase_blocks(track, [[0, 10, 30]], [[10, 30, 60]], np.array([0, 1, 2]), np.ones((60, 2)))
    np.testing.assert_array_equal(blocks["posture"][0, :, 0], [0, 4, 9, 10, 19, 29, 30, 44, 59])
    np.testing.assert_allclose(np.exp(blocks["tempo"]), [[10 / 60, 20 / 60, 30 / 60]])
    with pytest.raises(ValueError, match="contiguous"):
        phase_blocks(track, [[0, 11, 30]], [[10, 30, 60]], np.arange(3), np.ones((60, 2)))


def test_scaling_is_invariant_to_latent_channel_units_and_translation():
    rng = np.random.default_rng(3)
    b = dict(posture=rng.normal(size=(40, 9, 10)), movement=rng.normal(size=(40, 9, 2)),
             tempo=rng.normal(size=(40, 3)), semantic=np.zeros((40, 9), int))
    original = scaled_blocks(b, fit_scaling(b, np.arange(40)))
    b["posture"] = b["posture"] * np.geomspace(.01, 100, 10) + 13
    changed = scaled_blocks(b, fit_scaling(b, np.arange(40)))
    np.testing.assert_allclose(changed["posture"], original["posture"], rtol=1e-10, atol=1e-10)
