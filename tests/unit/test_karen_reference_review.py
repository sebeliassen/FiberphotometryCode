"""Native-support and descriptive-ranking safeguards for the pooled review."""

import numpy as np
import pytest

from studies.karen_pff_vs_monomer.analysis.render_reference_review import (
    balance_sample,
    normalize_descriptors,
    verify_call,
)
from studies.sequence_viz.recreate_generator_videos import rank_exemplar_neighbors
from studies.sequence_viz.render_generator_review_grids import select_movie_sets


def test_review_requires_full_native_core_not_only_phase_samples():
    track = dict(frames=np.array([10, 11, 13, 14]), labels=np.array([0, 0, 1, 2]),
                 centroid=np.zeros((4, 2)), heading=np.zeros(4))
    with pytest.raises(ValueError, match="missing or excluded"):
        verify_call(track, dict(start_frame=10, end_frame_exclusive=15, observed_tokens="[0, 1, 2]"), np.arange(3))


def test_review_validates_grouped_order_while_retaining_raw_member_changes():
    track = dict(frames=np.arange(10, 15), labels=np.array([0, 1, 2, 2, 3]),
                 centroid=np.zeros((5, 2)), heading=np.zeros(5))
    call = dict(start_frame=10, end_frame_exclusive=15, observed_tokens="[0, 1, 2]")
    verify_call(track, call, np.array([0, 0, 1, 2]))
    with pytest.raises(ValueError, match="grouped call"):
        verify_call(track, dict(call, observed_tokens="[0, 2, 1]"), np.array([0, 0, 1, 2]))


def test_calibration_contributes_equal_number_per_mouse_despite_occurrence_imbalance():
    values = np.arange(100)[:, None]
    sample = balance_sample(values, np.r_[np.zeros(95), np.ones(5)], per_animal=10)
    assert sample.shape == (20, 1)
    assert (sample < 95).sum() == 10
    assert (sample >= 95).sum() == 10
    np.testing.assert_array_equal(sample, balance_sample(values, np.r_[np.zeros(95), np.ones(5)], per_animal=10))


def test_finite_normalisation_keeps_different_executions_and_p80_direction():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(100, 449))
    normal, scales = normalize_descriptors(values, values)
    assert np.isfinite(normal).all() and scales["displacement_px_p75"] > 0
    d, p, exemplar = rank_exemplar_neighbors(normal, np.repeat(np.arange(10), 10))
    assert d[exemplar] == 0 and p[exemplar] == 100
    calls = [dict(recording=f"s{i}", subject=str(i % 10), start_frame=i) for i in range(100)]
    selected = select_movie_sets(calls, d, p, 1, exemplar)["p80"]
    assert len(selected) == 3 and len({c["recording"] for c in selected}) == 3
    assert all(abs(c["actual_percentile"] - 80) <= 5 for c in selected)


def test_robust_scales_preserve_equal_block_distance_in_native_feature_units():
    # Native latent/log-duration IQR=1.25, displacement P75=1px.
    # Construct already block-normalized descriptors, as the real extractor does.
    values = np.zeros((4, 449))
    levels = np.array([0., .5, 1.5, 2.])  # IQR exactly 1.25
    values[:, :320] = levels[:, None] / np.sqrt(4 * 320)
    values[:, 320:384] = np.tile([1., 0.], 32) / np.sqrt(4 * 64)
    values[:, -1] = levels / 2
    normalized, scales = normalize_descriptors(values, values)
    np.testing.assert_allclose(scales["latent_iqr"], np.full(10, 1.25))
    assert scales["displacement_px_p75"] == pytest.approx(1.)
    assert scales["log_duration_iqr"] == pytest.approx(1.25)
    # A 1-IQR change in every latent feature has exactly 1/4 squared distance,
    # independently of its 320 coordinates. Log duration gets the same weight.
    pose_change = np.zeros((1, 449))
    pose_change[:, :320] = 1.25 / np.sqrt(4 * 320)
    duration_change = np.zeros((1, 449))
    duration_change[:, -1] = 1.25 / 2
    pose, _ = normalize_descriptors(pose_change, values)
    duration, _ = normalize_descriptors(duration_change, values)
    assert np.sum(pose ** 2) == pytest.approx(.25)
    assert np.sum(duration ** 2) == pytest.approx(.25)
    np.testing.assert_allclose(normalized[:, :320], values[:, :320] / 1.25)


def test_latent_unit_rescaling_does_not_change_normalized_distance():
    rng = np.random.default_rng(23)
    calibration = rng.normal(size=(40, 449))
    values = rng.normal(size=(3, 449))
    normal, _ = normalize_descriptors(values, calibration)
    channel_units = np.tile(np.geomspace(.01, 100., 10), 32)
    calibration[:, :320] *= channel_units
    values[:, :320] *= channel_units
    rescaled, _ = normalize_descriptors(values, calibration)
    np.testing.assert_allclose(rescaled, normal, rtol=1e-12, atol=1e-12)
