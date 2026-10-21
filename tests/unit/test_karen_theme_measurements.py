"""Independent clock, branch and physical expectations for refined theme calls."""

import numpy as np
import pandas as pd
import pytest

from studies.karen_pff_vs_monomer.analysis.theme_measurements import (
    Kinematics,
    intervals_from_mask,
    matches,
    union_frames,
)


def test_stage_sets_keep_direction_variants_without_skipping_unknowns_or_gaps():
    table = pd.DataFrame(dict(token=[0, 6, 1, 7, 0, 22, 6, 0],
                              block=[0, 0, 0, 0, 0, 0, 1, 1],
                              start_frame=np.arange(8) * 2, end_frame_exclusive=np.arange(1, 9) * 2))
    assert matches(table, [[0, 1], [6, 7], [0, 1]]).tolist() == [0, 2]
    assert matches(table, [[6], [0]]).tolist() == [6]
    assert matches(table, [[0], [6]]).tolist() == [0]


def test_mask_and_missing_frames_are_real_barriers():
    assert intervals_from_mask([0, 1, 2, 4, 5, 6], [1, 0, 1, 1, 1, 0]).tolist() == [[0, 1], [2, 3], [4, 6]]
    assert union_frames([0, 2, 9], [4, 6, 10]) == 7
    with pytest.raises(ValueError, match="strictly increasing"):
        intervals_from_mask([1, 1], [1, 1])


def test_native_velocity_and_wrapped_heading_exclude_preceding_frame():
    cache = dict(frames=np.arange(5), fps=2., centroid=np.column_stack([np.arange(5) * 3., np.zeros(5)]),
                 heading=np.array([0., 3., -3., -2., -1.]), latent=np.column_stack([np.arange(5), np.arange(5)]))
    measured = Kinematics(cache, np.ones(5, bool)).measure([1], [4]).iloc[0]
    assert measured.duration_s == 1.5
    assert measured.mean_speed_px_s == 6.
    assert measured.net_displacement_px == measured.path_length_px == 6.
    assert measured.signed_heading_change_rad == pytest.approx(2 * np.pi - 5)
    assert measured.mean_abs_angular_velocity_rad_s == pytest.approx(2 * np.pi - 5)
    assert measured.latent_dispersion == pytest.approx(np.sqrt(2 / 3))


def test_unsupported_window_is_never_interpolated():
    cache = dict(frames=np.array([0, 1, 3, 4]), fps=60., centroid=np.zeros((4, 2)),
                 heading=np.zeros(4), latent=np.ones((4, 2)))
    with pytest.raises(ValueError, match="unsupported native"):
        Kinematics(cache, np.ones(4, bool)).measure([0], [5])
    with pytest.raises(ValueError, match="unsupported native"):
        Kinematics({**cache, "frames": np.arange(4)}, np.array([1, 0, 1, 1], bool)).measure([0], [3])


def test_one_frame_has_no_measured_pair_velocity():
    cache = dict(frames=np.arange(2), fps=60., centroid=np.zeros((2, 2)),
                 heading=np.zeros(2), latent=np.ones((2, 3)))
    row = Kinematics(cache, np.ones(2, bool)).measure([0], [1]).iloc[0]
    assert row.duration_s == 1 / 60
    assert np.isnan(row.mean_speed_px_s)
    assert row.latent_dispersion == 0
