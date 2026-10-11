"""Independent tiny examples of the preparation reused by sliding studies."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from fpcode.photometry.session_signals import legacy_nearest_window, prepare_region_traces


def recording():
    raw = pd.DataFrame({
        "SystemTimestamp": [100., 101., 103., 104.],
        "signal_0": [8., 12., np.nan, 20.],
        "signal_1": [10., np.nan, 14., 16.],
        "signal_2": [999., 999., 999., 999.],
        "signal_3": [90., 91., 92., 93.],
    })
    iso = pd.DataFrame({
        "SystemTimestamp": [100., 101., 103., 104.],
        "signal_0": [1., 2., 3., 4.],
        "signal_1": [2., 3., 4., 5.],
        "signal_2": [0., 0., 0., 0.],
        "signal_3": [1., 1., 1., 1.],
    })
    return SimpleNamespace(photometry={"470": raw, "415": iso}, metadata={"signal_meta": {
        "signal_1": {"region": "NAc"},
        "signal_0": {"region": "VTA"},
        "signal_2": {"region": "VTA"},
        "signal_3": {"region": "LH"},
        "absent_column": {"region": "PFC"},
    }})


def test_unnormalized_preparation_preserves_region_order_nan_positions_and_input_tables():
    session = recording()
    before = {channel: table.copy(deep=True) for channel, table in session.photometry.items()}
    calls = []

    def correct(raw, iso):
        calls.append((raw.copy(), iso.copy()))
        if raw[0] == 90:
            raise ValueError("invalid correction fit")
        return raw - 2 * iso

    prepared = prepare_region_traces(session, normalization="none", errors="skip", corrector=correct)
    assert list(prepared.traces) == ["NAc", "VTA"]
    np.testing.assert_allclose(prepared.traces["NAc"], [6, np.nan, 6, 6], equal_nan=True)
    np.testing.assert_allclose(prepared.traces["VTA"], [6, 8, np.nan, 12], equal_nan=True)
    np.testing.assert_array_equal(prepared.timestamps, [100, 101, 103, 104])
    assert prepared.failures["LH"] == "invalid correction fit"
    assert "absent_column" in prepared.failures["PFC"]
    assert len(calls) == 3  # One correction per region; repeated VTA uses its first column.
    for channel, table in before.items():
        pd.testing.assert_frame_equal(session.photometry[channel], table)


def test_prepared_outputs_own_copies_even_if_correction_returns_a_source_view():
    session = recording()
    before = {channel: table.copy(deep=True) for channel, table in session.photometry.items()}
    prepared = prepare_region_traces(session, normalization="none", errors="skip",
                                     corrector=lambda raw, iso: raw)
    prepared.timestamps[0] = -1
    prepared.traces["VTA"][0] = -1
    for channel, table in before.items():
        pd.testing.assert_frame_equal(session.photometry[channel], table)


def test_preparation_and_legacy_event_bounds_do_not_fill_missing_signal_samples():
    prepared = prepare_region_traces(recording(), normalization="none", errors="skip",
                                     corrector=lambda raw, iso: raw - 2 * iso)
    begin, end = legacy_nearest_window(prepared.timestamps, 101.5, 105)
    assert (begin, end) == (1, 3)
    np.testing.assert_allclose(prepared.traces["VTA"][begin:end], [8, np.nan], equal_nan=True)
    begin, end = legacy_nearest_window(prepared.timestamps, 101.5, 105, expected_length=3)
    assert (begin, end) == (1, 4)
    np.testing.assert_allclose(prepared.traces["VTA"][begin:end], [8, np.nan, 12], equal_nan=True)
