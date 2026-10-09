import numpy as np
import pytest

from fpcode.contracts import Bout, EventInterval, PoseSeries, SignalMetadata, SignalSeries, TimeAxis


@pytest.mark.parametrize("values", [[1, 1], [2, 1], [0, np.nan], [[1, 2]], ["one"]])
def test_time_axis_rejects_ambiguous_coordinates(values):
    with pytest.raises(ValueError):
        TimeAxis(values, "s", "acquisition")


def test_contracts_own_immutable_copies_and_validate_mask_shape():
    times, values = np.array([0, 1]), np.array([3., np.nan])
    signal = SignalSeries(TimeAxis(times, "s", "acquisition"), values, (SignalMetadata("VTA"),))
    times[0] = -10
    values[0] = 100
    np.testing.assert_array_equal(signal.time.values, [0, 1])
    np.testing.assert_allclose(signal.values, [3, np.nan], equal_nan=True)
    np.testing.assert_array_equal(signal.valid, [1, 0])
    with pytest.raises(ValueError):
        signal.values[0] = 9
    with pytest.raises(ValueError, match="validity"):
        SignalSeries(signal.time, signal.values, signal.signals, [True, True])


def test_pose_requires_explicit_spatial_frame_and_matching_confidence():
    axis = TimeAxis([0, 1], "frame", "video")
    pose = PoseSeries(axis, [[[0, 1]], [[2, 3]]], [[1], [.5]], ("nose",), "px", "arena")
    assert pose.coordinates.shape == (2, 1, 2)
    assert pose.valid.all()
    with pytest.raises(ValueError, match="confidence"):
        PoseSeries(axis, pose.coordinates, [[1]], ("nose",), "px", "arena")
    with pytest.raises(ValueError, match="coordinate frame"):
        PoseSeries(axis, pose.coordinates, pose.confidence, ("nose",), "px", "")


def test_events_and_bouts_define_half_open_bounds():
    assert Bout(3, 2, 5).duration == 3
    assert EventInterval("injection", 2, 5, "s", "acquisition").stop == 5
    with pytest.raises(ValueError):
        Bout(3, 5, 2)
    with pytest.raises(ValueError):
        EventInterval("bad", 5, 2, "s", "acquisition")
