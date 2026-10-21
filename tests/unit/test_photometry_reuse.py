from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fpcode.adapters.legacy_sessions import as_signal_series
from fpcode.photometry import calculate_signal_response_metrics_matrix, group_signals, preprocessing
from fpcode.photometry.session_signals import legacy_nearest_window, prepare_region_traces
from fpcode.statistics import compute_mean_ci
from studies.photometry_exploration.analysis import response_metrics, signal_processing


def session():
    table = pd.DataFrame({"SystemTimestamp": [10, 11, 12], "signal_0": [1., 2., 3.], "signal_1": [4., 5., 6.]})
    return SimpleNamespace(
        photometry={"470": table, "415": table.copy()},
        metadata={"signal_meta": {"signal_0": {"region": "VTA"}, "signal_1": {"region": "NAc"}}},
    )


def test_grouping_shares_one_implementation_and_preserves_missing_policy():
    s = SimpleNamespace(signal_info={"a": {"signal_matrix": np.array([[1, 2]])}, "empty": {"signal_matrix": np.empty((0, 2))}})
    key = lambda record, name: name
    np.testing.assert_array_equal(group_signals([s, s], key)["a"], [[1, 2], [1, 2]])
    np.testing.assert_array_equal(signal_processing.group_signals([s, s], key)["a"], [[1, 2], [1, 2]])
    assert group_signals([SimpleNamespace()], key) == {}
    with pytest.raises(AttributeError):
        signal_processing.group_signals([SimpleNamespace()], key)


def test_summary_values_preserve_population_sd_and_empty_policy():
    mean, low, high = compute_mean_ci([[0, 2], [2, 6]])
    np.testing.assert_array_equal(mean, [1, 4])
    width = 1.959963984540054 * np.array([1, 2]) / np.sqrt(2)
    np.testing.assert_allclose(low, mean - width)
    np.testing.assert_allclose(high, mean + width)
    for expected, actual in zip((mean, low, high), signal_processing.compute_mean_ci([[0, 2], [2, 6]])):
        np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(compute_mean_ci(np.empty((0, 2)))[0], [0, 0])
    with pytest.warns(RuntimeWarning):
        assert np.isnan(signal_processing.compute_mean_ci(np.empty((0, 2)))[0]).all()


def test_response_values_and_exploratory_tuple_api():
    signals = np.array([[0., 2., 1.], [1., 3., 1.]])
    expected = [[4, 1, 2, .5, 2.5], [4, 2, 3, .5, 4]]
    np.testing.assert_array_equal(calculate_signal_response_metrics_matrix(signals, fps=2), expected)
    matrix, names = response_metrics.calculate_signal_response_metrics_matrix(signals, None, 2)
    np.testing.assert_array_equal(matrix, expected)
    assert names == {0: "slope_up", 1: "slope_down", 2: "maximal_value", 3: "peak_timing", 4: "auc"}
    scalar = response_metrics.calculate_signal_response_metrics(signals[0], None)
    assert scalar == dict(slope_up=2., slope_down=.5, maximal_value=2., peak_timing=1., auc=2.5)
    with pytest.raises(ValueError):
        response_metrics.calculate_signal_response_metrics(np.array([]), None)


def test_record_adapter_requires_units_and_keeps_original_data():
    s = session()
    before = s.photometry["470"].copy(deep=True)
    signals = as_signal_series(s, "470", time_unit="s", clock="photometer")
    assert [label.region for label in signals.signals] == ["VTA", "NAc"]
    np.testing.assert_array_equal(signals.time.values, [10, 11, 12])
    pd.testing.assert_frame_equal(s.photometry["470"], before)


def test_shared_preparation_keeps_first_region_column_and_reports_failures():
    s = session()
    s.metadata["signal_meta"]["signal_1"]["region"] = "VTA"
    calls = []
    def correction(raw, iso):
        calls.append(raw.copy())
        return raw - iso + np.array([0., 2., 4.])
    result = prepare_region_traces(s, corrector=correction)
    assert len(calls) == 1
    np.testing.assert_array_equal(result.traces["VTA"], [0, 2, 4])
    normalized = prepare_region_traces(s, normalization="global_zscore", corrector=correction)
    np.testing.assert_allclose(normalized.traces["VTA"], [-np.sqrt(1.5), 0, np.sqrt(1.5)])
    assert result.failures == {}
    def failing(raw, iso):
        raise ValueError("bad fit")
    skipped = prepare_region_traces(s, errors="skip", corrector=failing)
    assert skipped.traces == {} and skipped.failures == {"VTA": "bad fit"}
    with pytest.raises(ValueError, match="bad fit"):
        prepare_region_traces(s, corrector=failing)
    np.testing.assert_array_equal(s.photometry["470"].signal_0, [1, 2, 3])


def test_legacy_window_bounds_ties_and_requested_row_length():
    assert legacy_nearest_window([0, 2, 4], 1, 3) == (0, 1)
    assert legacy_nearest_window([0, 2, 4], -10, 20) == (0, 2)
    assert legacy_nearest_window([0, 2, 4], 0, 2, expected_length=2) == (0, 2)


def test_exponential_fit_and_ols_correction_preserve_nan_positions(monkeypatch):
    monkeypatch.setattr(preprocessing, "curve_fit", lambda *args, **kwargs: (np.array([0., 1., 0., 1., 5.]), None))
    raw = np.array([5., 6., np.nan, 8.])
    np.testing.assert_allclose(preprocessing.fit_biexponential_dff(raw), [0, .2, np.nan, .6], equal_nan=True)
    np.testing.assert_allclose(raw, [5, 6, np.nan, 8], equal_nan=True)
    monkeypatch.setattr(preprocessing, "fit_biexponential_dff", lambda value: np.asarray(value).copy())
    corrected = preprocessing.motion_correct_biexponential(np.array([1., 3., 5.]), np.array([0., 1., 2.]))
    np.testing.assert_allclose(corrected, [0, 0, 0], atol=1e-12)
