import h5py
import numpy as np
import pandas as pd
import pytest

from fpcode.adapters.acquisition import read_photometry_csv
from fpcode.adapters.dlc import pose_from_dlc_table
from fpcode.adapters.kpms_results import read_kpms_results
from fpcode.alignment.clocks import ClockTransform
from fpcode.contracts import SignalMetadata, SignalSeries, TimeAxis
from fpcode.contracts.sessions import SessionKey
from fpcode.tracking.registration import fit_affine
from fpcode.workflows.multimodal_analysis import MultimodalSession, assemble_multimodal_session


def test_actual_small_csv_hdf_and_dlc_modalities_keep_gaps_and_values(tmp_path):
    acquisition = tmp_path / "photometry.csv"
    pd.DataFrame({"time": [10, 11, 12, 13], "signal": [0, 2, 4, 6]}).to_csv(acquisition, index=False)
    signal = read_photometry_csv(acquisition, time_column="time", signal_columns=["signal"],
                                 time_unit="s", clock="hardware")
    results_path = tmp_path / "results.h5"
    with h5py.File(results_path, "w") as stream:
        stream["recording/syllable"] = [0, 1]
    results = read_kpms_results(results_path)["recording"]
    assert not results["syllable"].flags.writeable
    time = TimeAxis([0, 1, 2, 3], "s", "video")
    columns = pd.MultiIndex.from_product([["dlc"], ["nose"], ["x", "y", "likelihood"]])
    table = pd.DataFrame([[0, 1, 0.9]] * 4, columns=columns)
    pose = pose_from_dlc_table(table, time=time, unit="px", coordinate_frame="top")
    transform = ClockTransform("hardware", "video", "s", "s", 10, 0, 1)
    session = assemble_multimodal_session(SessionKey("recording"), signal, time,
                                         model_labels=results["syllable"], original_frames=[0, 2],
                                         target_frames=[0, 1, 2, 3], pose=pose, clock_transform=transform)
    np.testing.assert_array_equal(session.signals.values[:, 0], [0, 2, 4, 6])
    np.testing.assert_array_equal(session.labels, [0, -1, 1, -1])
    np.testing.assert_array_equal(session.label_valid, [True, False, True, False])
    assert not session.labels.flags.writeable
    np.testing.assert_array_equal(table.to_numpy(), [[0, 1, 0.9]] * 4)


def test_clock_calibration_preserves_integer_nanosecond_offsets():
    origin = 10**18
    axis = TimeAxis(np.array([origin, origin + 1, origin + 4], dtype=np.int64), "ns", "hardware")
    transformed = ClockTransform("hardware", "video", "ns", "s", origin, 0, 1e-9).apply(axis)
    np.testing.assert_allclose(transformed.values, [0, 1e-9, 4e-9], rtol=0, atol=1e-24)
    with pytest.raises(ValueError, match="does not match"):
        ClockTransform("other", "video", "ns", "s", origin, 0, 1e-9).apply(axis)


def test_affine_registration_has_independent_expected_coordinates():
    source = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    target = np.array([[3, 4], [5, 4], [3, 7], [5, 7]])
    registration = fit_affine(source, target)
    np.testing.assert_allclose(registration.apply([[2, 2]]), [[7, 10]], atol=1e-12)
    assert registration.residual_rms < 1e-12
    with pytest.raises(ValueError, match="collinear"):
        fit_affine([[0, 0], [1, 0], [2, 0]], [[0, 0], [1, 0], [2, 0]])


@pytest.mark.parametrize("axis", [TimeAxis([0, 1], "s", "other"), TimeAxis([0, 1], "ms", "video"),
                                 TimeAxis([1, 2], "s", "video"), TimeAxis([0], "s", "video")])
def test_direct_multimodal_contract_rejects_incoherent_signal_timelines(axis):
    signal = SignalSeries(axis, np.arange(len(axis)), (SignalMetadata("a"),))
    with pytest.raises(ValueError, match="signals must use the session clock and timeline"):
        MultimodalSession(SessionKey("recording"), TimeAxis([0, 1], "s", "video"), signal,
                          np.array([0, 1]), np.array([True, True]))


@pytest.mark.parametrize("labels,valid", [([0., 1.], [True, True]), ([0, -1], [True, True])])
def test_direct_multimodal_contract_rejects_invalid_label_claims(labels, valid):
    axis = TimeAxis([0, 1], "s", "video")
    signal = SignalSeries(axis, [3, 7], (SignalMetadata("a"),))
    with pytest.raises(ValueError, match="nonnegative integers"):
        MultimodalSession(SessionKey("recording"), axis, signal, np.array(labels), np.array(valid))
