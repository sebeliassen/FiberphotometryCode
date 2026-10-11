import numpy as np
import pytest

from fpcode.contracts import EventInterval, SignalMetadata, SignalSeries, TimeAxis
from fpcode.photometry.event_windows import extract_event_windows


def test_event_windows_preserve_coordinates_masks_and_channels():
    signal = SignalSeries(TimeAxis([0, 1, 2], "s", "recording"), [[0, 10], [2, 12], [4, 14]],
                          (SignalMetadata("a"), SignalMetadata("b")))
    events = [EventInterval("one", 1, 2, "s", "recording"), EventInterval("two", 2, 3, "s", "recording")]
    windows = extract_event_windows(signal, events, [-1, 0, 1])
    np.testing.assert_array_equal(windows.values[0], [[0, 10], [2, 12], [4, 14]])
    np.testing.assert_allclose(windows.values[1], [[2, 12], [4, 14], [np.nan, np.nan]], equal_nan=True)
    np.testing.assert_array_equal(windows.valid[1], [[1, 1], [1, 1], [0, 0]])
    assert not windows.values.flags.writeable
    assert windows.unit == "s" and windows.clock == "recording"
    assert extract_event_windows(signal, [], [-1, 0, 1]).values.shape == (0, 3, 2)
    with pytest.raises(ValueError, match="units and clock"):
        extract_event_windows(signal, [EventInterval("bad", 1, 2, "ms", "recording")], [0])


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("offsets", [[-1, 0, 1], [-1., 0., 1.]])
def test_event_windows_add_large_nanosecond_origins_without_float_promotion(dtype, offsets):
    origin = 10**18
    signal = SignalSeries(TimeAxis(np.array([origin - 1, origin, origin + 1], dtype=dtype), "ns", "hardware"),
                          [3, 7, 11], (SignalMetadata("a"),))
    windows = extract_event_windows(signal, [EventInterval("event", origin, origin + 1, "ns", "hardware")],
                                    offsets)
    np.testing.assert_array_equal(windows.values[0, :, 0], [3, 7, 11])
    assert windows.valid.all()


def test_event_windows_preserve_negative_offsets_with_unsigned_coordinates():
    signal = SignalSeries(TimeAxis(np.array([0, 1, 2], dtype=np.uint64), "ns", "hardware"),
                          [3, 7, 11], (SignalMetadata("a"),))
    windows = extract_event_windows(signal, [EventInterval("event", 1, 2, "ns", "hardware")], [-2., -1., 0.])
    np.testing.assert_allclose(windows.values[0, :, 0], [np.nan, 3, 7], equal_nan=True)
    np.testing.assert_array_equal(windows.valid[0, :, 0], [False, True, True])
