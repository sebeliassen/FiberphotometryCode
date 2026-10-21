"""Small exact numerical examples of strict and historical alignment policies."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fpcode.alignment import align_signal, remap_model_frames
from fpcode.alignment.clocks import ClockTransform
from fpcode.contracts import SignalMetadata, SignalSeries, TimeAxis
from fpcode.photometry.alignment import build_fixed_moseq_dfs, combined_photometry_and_syllables


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


@pytest.mark.parametrize("source_dtype,target_dtype", [(np.uint64, np.int64), (np.int64, np.uint64)])
def test_large_nanosecond_alignment_preserves_mixed_integer_coordinates(source_dtype, target_dtype):
    origin = 10**18
    source = series(np.array([origin, origin + 2, origin + 4], dtype=source_dtype), [3, 7, 11], unit="ns")
    target = TimeAxis(np.array([origin - 1, origin, origin + 1, origin + 2, origin + 4, origin + 5],
                              dtype=target_dtype), "ns", "camera")
    expected_exact = [np.nan, 3, np.nan, 7, 11, np.nan]
    expected_nearest = [np.nan, 3, 3, 7, 11, np.nan]
    expected_linear = [np.nan, 3, 5, 7, 11, np.nan]
    for method, policy, expected in [
        ("exact", {}, expected_exact),
        ("nearest", {"tolerance": 1}, expected_nearest),
        ("linear", {"max_gap": 2}, expected_linear),
    ]:
        aligned = align_signal(source, target, method=method, **policy)
        np.testing.assert_allclose(aligned.values, expected, equal_nan=True)
        np.testing.assert_array_equal(aligned.valid, np.isfinite(expected))
        np.testing.assert_array_equal(aligned.time.values, target.values)


def test_integer_alignment_avoids_signed_subtraction_overflow():
    origin = np.iinfo(np.int64).min
    source = series(np.array([origin, 0], dtype=np.int64), [0, 2], unit="ns")
    target = TimeAxis(np.array([origin // 2], dtype=np.int64), "ns", "camera")
    np.testing.assert_array_equal(align_signal(source, target, method="linear", max_gap=2**63).values, [1])
    np.testing.assert_array_equal(align_signal(source, target, method="nearest", tolerance=2**62).values, [0])


def test_integer_alignment_supports_a_span_larger_than_uint64():
    source = series(np.array([-1, 0], dtype=np.int64), [1, 2], unit="ns")
    target = TimeAxis(np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64), "ns", "camera")
    aligned = align_signal(source, target, method="nearest", tolerance=0)
    np.testing.assert_allclose(aligned.values, [2, np.nan], equal_nan=True)


@pytest.mark.parametrize("method,policy", [("exact", {}), ("nearest", {"tolerance": 2}),
                                          ("linear", {"max_gap": 2})])
def test_integral_float_target_before_large_integer_source_is_not_an_exact_match(method, policy):
    origin = 10**18
    source = series(np.array([origin + 1, origin + 2], dtype=np.int64), [3, 7], unit="ns")
    target = TimeAxis(np.array([float(origin)]), "ns", "camera")
    aligned = align_signal(source, target, method=method, **policy)
    assert np.isnan(aligned.values[0]) and not aligned.valid[0]


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("float_origin", [False, True])
def test_clock_transform_subtracts_integer_timestamps_before_float_conversion(dtype, float_origin):
    origin = 10**18
    source = TimeAxis(np.array([origin - 2, origin, origin + 1], dtype=dtype), "ns", "hardware")
    calibration_origin = float(origin) if float_origin else origin
    transform = ClockTransform("hardware", "video", "ns", "s", calibration_origin, 0, 1e-9)
    np.testing.assert_allclose(transform.apply(source).values, [-2e-9, 0, 1e-9], rtol=0, atol=1e-24)


def test_clock_transform_preserves_fractional_origin_and_unsigned_negative_offsets():
    source = TimeAxis(np.array([0, 2, 4], dtype=np.uint64), "ns", "hardware")
    transform = ClockTransform("hardware", "video", "ns", "ns", 2.5, 0, 1)
    np.testing.assert_array_equal(transform.apply(source).values, [-2.5, -.5, 1.5])


def test_clock_transform_does_not_round_an_integer_origin_to_the_float_axis():
    origin = 10**18
    source = TimeAxis(np.array([float(origin), float(origin + 128)]), "ns", "hardware")
    transform = ClockTransform("hardware", "video", "ns", "ns", origin + 1, 0, 1)
    np.testing.assert_array_equal(transform.apply(source).values, [-1, 127])


def test_clock_transform_does_not_overflow_a_signed_origin_subtraction():
    source = TimeAxis(np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).min + 4096], dtype=np.int64),
                      "ns", "hardware")
    transform = ClockTransform("hardware", "video", "ns", "ns", 4096, 0, 1.5)
    # The first rebased value is below int64 minimum, so signed arithmetic
    # would wrap. These endpoints remain distinct in the floating output.
    np.testing.assert_array_equal(transform.apply(source).values,
                                  [float(-2**63 - 4096) * 1.5, float(-2**63) * 1.5])


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_clock_transform_preserves_integer_target_epochs(scale, dtype):
    origin = 10**18
    source = TimeAxis(np.array([origin, origin + 1, origin + 2], dtype=dtype), "ns", "hardware")
    transform = ClockTransform("hardware", "video", "ns", "ns", origin, origin, scale)
    result = transform.apply(source)
    np.testing.assert_array_equal(result.values, [origin, origin + scale, origin + 2 * scale])
    assert np.issubdtype(result.values.dtype, np.integer)


def test_fractional_clock_origin_can_map_to_integral_target_coordinates():
    origin = 10**18
    source = TimeAxis([0, 1, 2], "ns", "hardware")
    transform = ClockTransform("hardware", "video", "ns", "ns", .5, origin, 2)
    np.testing.assert_array_equal(transform.apply(source).values, [origin - 1, origin + 1, origin + 3])


def test_fractional_target_offsets_at_large_epochs_require_rebasing():
    source = TimeAxis([0, 1, 2], "ns", "hardware")
    transform = ClockTransform("hardware", "video", "ns", "ns", 0, 10**18, .5)
    with pytest.raises(ValueError, match="rebasing the target clock"):
        transform.apply(source)


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
