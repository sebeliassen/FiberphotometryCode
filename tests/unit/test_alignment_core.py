"""Small exact numerical examples of strict and historical alignment policies."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fpcode.alignment import align_signal, remap_model_frames
from fpcode.contracts import SignalMetadata, SignalSeries, TimeAxis
from fpcode.photometry.alignment import combined_photometry_and_syllables, build_fixed_moseq_dfs


def series(times, values, valid=None, unit="s"):
    values = np.asarray(values)
    labels = tuple(SignalMetadata(str(i)) for i in range(1 if values.ndim == 1 else values.shape[1]))
    return SignalSeries(TimeAxis(times, unit, "camera"), values, labels, valid)


def test_strict_multichannel_linear_values_masks_and_no_endpoint_fill():
    source = series([0, 2, 8], [[0, 10], [4, 14], [16, 26]], [[1, 1], [1, 0], [1, 1]])
    target = TimeAxis([-1, 0, 1, 2, 5, 8, 9], "s", "camera")
    result = align_signal(source, target, method="linear", max_gap=3)
    np.testing.assert_allclose(result.values[:, 0], [np.nan, 0, 2, 4, np.nan, 16, np.nan], equal_nan=True)
    np.testing.assert_allclose(result.values[:, 1], [np.nan, 10, np.nan, np.nan, np.nan, 26, np.nan], equal_nan=True)
    np.testing.assert_array_equal(result.valid[:, 1], [0, 1, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(source.values, [[0, 10], [4, 14], [16, 26]])


def test_exact_nearest_ties_and_nanosecond_precision():
    source = series([0, 2], [3, 7])
    target = TimeAxis([0, 1, 2], "s", "camera")
    np.testing.assert_allclose(align_signal(source, target).values, [3, np.nan, 7], equal_nan=True)
    np.testing.assert_array_equal(align_signal(source, target, method="nearest", tolerance=1).values, [3, 3, 7])
    np.testing.assert_allclose(align_signal(source, target, method="nearest", tolerance=.1).values, [3, np.nan, 7], equal_nan=True)
    ns = series([10**18, 10**18 + 2], [0, 2], unit="ns")
    target_ns = TimeAxis([10**18 + 1], "ns", "camera")
    np.testing.assert_array_equal(align_signal(ns, target_ns, method="linear", max_gap=2).values, [1])


@pytest.mark.parametrize("kwargs", [{"method": "nearest"}, {"method": "linear"}, {"method": "linear", "max_gap": 0}, {"method": "other"}])
def test_alignment_requires_explicit_supported_policies(kwargs):
    with pytest.raises(ValueError):
        align_signal(series([0, 2], [0, 1]), TimeAxis([1], "s", "camera"), **kwargs)


def test_alignment_rejects_unit_and_clock_mismatch():
    source = series([0, 2], [0, 1])
    for axis in [TimeAxis([1], "ms", "camera"), TimeAxis([1], "s", "photometer")]:
        with pytest.raises(ValueError, match="units and clock"):
            align_signal(source, axis)


def test_legacy_multichannel_duplicate_first_and_endpoint_fill_are_preserved():
    phot = {
        "470": pd.DataFrame({"frame_idx": [0, 0, 2], "SystemTimestamp": [10, 99, 12], "signal_0": [0, 99, 2]}),
        "415": pd.DataFrame({"frame_idx": [0, 2], "SystemTimestamp": [10, 12], "signal_0": [20, 24]}),
    }
    snapshots = {k: v.copy(deep=True) for k, v in phot.items()}
    session = SimpleNamespace(syllables=pd.DataFrame(index=[0, 1, 2, 3]), photometry=phot)
    result = combined_photometry_and_syllables(session, freqs=["470", "415"])
    np.testing.assert_array_equal(result.signal_0_470, [0, 1, 2, 2])
    np.testing.assert_array_equal(result.signal_0_415, [20, 22, 24, 24])
    for key in phot:
        pd.testing.assert_frame_equal(phot[key], snapshots[key])


def test_row_interpolation_legacy_and_coordinate_interpolation_are_distinct():
    session = SimpleNamespace(
        syllables=pd.DataFrame(index=[0, 1, 10]),
        photometry={"470": pd.DataFrame({"frame_idx": [0, 10], "SystemTimestamp": [0, 10], "signal_0": [0, 100]})},
    )
    np.testing.assert_array_equal(combined_photometry_and_syllables(session).signal_0, [0, 50, 100])
    strict = align_signal(series([0, 10], [0, 100]), TimeAxis([0, 1, 10], "s", "camera"), method="linear", max_gap=10)
    np.testing.assert_array_equal(strict.values, [0, 10, 100])


def test_legacy_alignment_missing_columns_and_unknown_policy():
    session = SimpleNamespace(syllables=pd.DataFrame(index=[0]), photometry={"470": pd.DataFrame({"frame_idx": [0]})})
    with pytest.raises(KeyError, match="SystemTimestamp"):
        combined_photometry_and_syllables(session)
    with pytest.raises(ValueError, match="unsupported"):
        combined_photometry_and_syllables(session, policy="linear")


def test_remapping_uses_model_coordinates_and_keeps_input_unchanged():
    model = pd.DataFrame({"frame_index": [2, 0], "syllable": [9, 3]})
    snapshot = model.copy(deep=True)
    result = remap_model_frames(model, np.array([10, 15, 20]))
    np.testing.assert_array_equal(result.frame_index, [20, 10])
    pd.testing.assert_frame_equal(model, snapshot)
    with pytest.raises(ValueError, match="strictly increasing"):
        remap_model_frames(model, np.array([10, 10, 20]))
    with pytest.raises(ValueError, match="outside"):
        remap_model_frames(model, np.array([10]))


def test_historical_npz_remapping_is_by_row_order(tmp_path):
    (tmp_path / "model").mkdir()
    np.savez(tmp_path / "model" / "video_frame_indexes.npz", session=[10, 20])
    model = pd.DataFrame({"name": ["session", "session"], "frame_index": [1, 0], "syllable": [9, 3]})
    before = model.copy()
    sessions = [SimpleNamespace(results_dict_key="session")]
    out = build_fixed_moseq_dfs(sessions, model, tmp_path, "model")
    np.testing.assert_array_equal(out[0].frame_index, [10, 20])
    pd.testing.assert_frame_equal(model, before)
    with pytest.raises(ValueError, match="Length mismatch"):
        build_fixed_moseq_dfs(sessions, model.iloc[:1], tmp_path, "model")
