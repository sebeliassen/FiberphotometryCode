"""Independent values for the explicit historical tandem motion policy."""

import numpy as np
import pytest

from fpcode.behavior.kinematics import legacy_nose_tail_kinematics


def test_forward_velocity_final_step_and_clipped_median_are_preserved():
    centers = np.array([[0., 0.], [1., 0.], [11., 0.], [12., 0.]])
    nose, tail = centers + [1., 0.], centers - [1., 0.]
    before = nose.copy(), tail.copy()
    result = legacy_nose_tail_kinematics(nose, tail)
    np.testing.assert_array_equal(result["centroid"], centers)
    np.testing.assert_array_equal(result["velocity_raw"], [60, 600, 60, 60])
    np.testing.assert_array_equal(result["velocity"], [60, 60, 60, 60])
    np.testing.assert_array_equal(result["heading"], [0, 0, 0, 0])
    np.testing.assert_array_equal(nose, before[0])
    np.testing.assert_array_equal(tail, before[1])


def test_angular_velocity_unwraps_the_phase_boundary_before_differentiating():
    angles = np.deg2rad([170., -170., -150.])
    nose = np.column_stack([np.cos(angles), np.sin(angles)])
    result = legacy_nose_tail_kinematics(nose, -nose)
    np.testing.assert_allclose(result["angular_velocity"], [20 * np.pi / 3] * 3)
    np.testing.assert_array_equal(result["velocity_raw"], [0, 0, 0])


@pytest.mark.parametrize("frames", [0, 1])
def test_forward_differences_reject_sequences_without_a_motion_step(frames):
    with pytest.raises(ValueError, match="at least two frames"):
        legacy_nose_tail_kinematics(np.zeros((frames, 2)), np.zeros((frames, 2)))
