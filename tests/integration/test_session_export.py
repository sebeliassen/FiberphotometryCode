"""Real tiny HDF/pickle export with independent legacy scientific expectations."""

import os
import pickle

import h5py
import numpy as np
import pandas as pd
import pytest

from fpcode.adapters.kpms_results import open_kpms_results, results_to_frame_table
from fpcode.kpms.export import export_tandem_cohort_results
from fpcode.workflows import session_export
from syllable_generation.datamodel import SessionRecord


def write_pose(path, nose_name, centers, offset):
    centers, offset = np.asarray(centers, dtype=float), np.asarray(offset, dtype=float)
    columns = {}
    for name, coordinates in ((nose_name, centers + offset), ("tail_base", centers - offset)):
        columns[("scorer", name, "x")] = coordinates[:, 0]
        columns[("scorer", name, "y")] = coordinates[:, 1]
        columns[("scorer", name, "likelihood")] = np.ones(len(centers))
    table = pd.DataFrame(columns)
    table.columns = pd.MultiIndex.from_tuples(table.columns, names=["scorer", "bodyparts", "coords"])
    table.to_hdf(path, key="df_with_missing", mode="w", format="fixed")


@pytest.fixture
def export_inputs(tmp_path):
    top, bot, result = (tmp_path / name for name in ("top.h5", "bot.h5", "results.h5"))
    # Fifth tracking frame is intentionally outside the four-frame model prefix.
    write_pose(top, "nose", [[0, 0], [1, 0], [3, 0], [6, 0], [99, 99]], [1, 0])
    write_pose(bot, "nose_tip", [[0, 10], [0, 12], [0, 14], [0, 16], [99, 99]], [0, 1])
    with h5py.File(result, "w") as handle:
        for sid in ("phot", "missing"):
            group = handle.create_group(f"{sid}_tandem_23kp")
            group["syllable"] = [0, 0, 1, 1]
            group["latent_state"] = np.arange(8).reshape(4, 2) / 10
            group["centroid"] = np.tile([30., 40.], (4, 1))
            group["heading"] = [.25, .5, .75, 1.]
    phot = SessionRecord("phot", project="fixture", metadata={"keep": "metadata"})
    phot.photometry["470"] = pd.DataFrame({
        "frame_idx": [0, 0, 3], "SystemTimestamp": [100., 999., 103.],
        "signal_0": [10., 99., 40.], "signal_0_zscore": [0., 99., 3.],
    })
    missing = SessionRecord("missing")
    missing.photometry["470"] = pd.DataFrame({"signal_0": [1.]})  # Explicit fallback for malformed input.
    sessions = [phot, missing, SessionRecord("skipped"), SessionRecord("unpaired")]
    dirty, reference = tmp_path / "dirty.pkl", tmp_path / "reference.pkl"
    dirty.write_bytes(pickle.dumps(sessions))
    ref = SessionRecord("phot")
    ref.all_data_df = pd.DataFrame({"syllable": [7, 8, 9, 10, 99]})
    reference.write_bytes(pickle.dumps([ref]))
    pairs = [{"sid": sid, "top_h5": str(top), "bot_h5": str(bot)}
             for sid in ("phot", "missing", "skipped")]
    sources = [top, bot, result, dirty, reference]
    return {"results": result, "dirty": dirty, "reference": reference, "pairs": pairs,
            "source_bytes": {path: path.read_bytes() for path in sources}}


def test_export_roundtrip_preserves_classes_columns_prefixes_and_legacy_alignment(tmp_path, export_inputs):
    inputs = export_inputs
    output, second = tmp_path / "published/sessions.pkl", tmp_path / "data/sessions.pkl"
    summary = export_tandem_cohort_results(inputs["results"], inputs["dirty"], output,
                                          second, inputs["reference"], pairs=inputs["pairs"])
    assert summary["total_sessions"] == 2 and summary["skipped_sessions"] == ["skipped"]
    assert summary["photometry_present"] == 1 and summary["photometry_nan"] == 1
    assert output.read_bytes() == second.read_bytes()
    modeled, absent = pickle.loads(output.read_bytes())
    assert type(modeled) is SessionRecord and type(absent) is SessionRecord
    assert modeled.project == "fixture" and modeled.metadata == {"keep": "metadata"}
    assert modeled.results_dict_key == "phot_tandem_23kp"
    assert modeled.syllables.columns.tolist() == [
        "syllable", "latent_state_0", "latent_state_1", "centroid_0", "centroid_1", "heading"]
    assert str(modeled.syllables.syllable.dtype) == "Int64"
    np.testing.assert_array_equal(modeled.syllables.latent_state_1, [.1, .3, .5, .7])
    table = modeled.all_data_df
    assert str(table.syllable.dtype) == str(table.tandem_syllable.dtype) == "Int64"
    np.testing.assert_array_equal(table.frame_index, [0, 1, 2, 3])
    np.testing.assert_array_equal(table.onset, [True, False, True, False])
    np.testing.assert_array_equal(table.centroid_x, [0, 1, 3, 6])
    np.testing.assert_array_equal(table.centroid_y_bot, [10, 12, 14, 16])
    np.testing.assert_array_equal(table.velocity_px_s_raw, [60, 120, 180, 180])
    np.testing.assert_array_equal(table.velocity_px_s, [60, 120, 180, 180])
    np.testing.assert_array_equal(table.velocity_px_s_bot, [120, 120, 120, 120])
    np.testing.assert_array_equal(table.heading_top, [0, 0, 0, 0])
    np.testing.assert_allclose(table.heading_bot, np.pi / 2)
    np.testing.assert_array_equal(table.heading, [.25, .5, .75, 1.])
    np.testing.assert_array_equal(table.angular_velocity_top, [0, 0, 0, 0])
    np.testing.assert_array_equal(table.top_syllable, [7, 8, 9, 10])
    np.testing.assert_array_equal(table.SystemTimestamp, [100, 101, 102, 103])
    np.testing.assert_array_equal(table.signal_0, [10, 20, 30, 40])
    np.testing.assert_array_equal(table.signal_0_zscore, [0, 1, 2, 3])
    assert table.columns[-25:-20].tolist() == ["SystemTimestamp", "signal_0", "signal_1", "signal_2", "signal_3"]
    assert table.columns[-1] == "signal_3_prominence" and table.signal_3_prominence.isna().all()
    assert absent.all_data_df.iloc[:, -25:].isna().all().all()
    assert not list(output.parent.glob(".*.tmp"))
    assert {path: path.read_bytes() for path in inputs["source_bytes"]} == inputs["source_bytes"]


def test_streamed_result_reader_owns_immutable_arrays_and_projects_nullable_labels(export_inputs):
    with open_kpms_results(export_inputs["results"]) as reader:
        assert "phot_tandem_23kp" in reader and "unknown" not in reader
        record = reader["phot_tandem_23kp"]
        assert all(not value.flags.writeable for value in record.values())
        table = results_to_frame_table(record, index=pd.Index([5, 7, 11, 13]))
    # Returned arrays and tables remain usable after the HDF scope closes.
    assert table.index.tolist() == [5, 7, 11, 13]
    assert table.syllable.tolist() == [0, 0, 1, 1]
    assert str(table.syllable.dtype) == "Int64"
    table.loc[5, "centroid_0"] = -1
    assert record["centroid"][0, 0] == 30


@pytest.mark.parametrize("alias", ["dirty", "results", "tracking", "hardlink", "duplicate_output"])
def test_export_rejects_source_and_duplicate_output_aliases_before_publication(tmp_path, export_inputs, alias):
    inputs = export_inputs
    output = tmp_path / "output.pkl"
    secondary = None
    if alias in {"dirty", "results"}:
        output = inputs[alias]
    elif alias == "tracking":
        output = inputs["pairs"][0]["top_h5"]
    elif alias == "hardlink":
        os.link(inputs["dirty"], output)
    else:
        secondary = output
    with pytest.raises(ValueError, match="aliases an input|must be distinct"):
        export_tandem_cohort_results(inputs["results"], inputs["dirty"], output,
                                     secondary, pairs=inputs["pairs"])
    assert {path: path.read_bytes() for path in inputs["source_bytes"]} == inputs["source_bytes"]
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("stage", ["serialization", "secondary_copy"])
def test_failed_export_staging_preserves_both_existing_outputs(tmp_path, monkeypatch, stage):
    outputs = [tmp_path / "first.pkl", tmp_path / "second.pkl"]
    for path in outputs:
        path.write_bytes(b"previous accepted output")

    def fail_dump(value, stream):
        stream.write(b"incomplete serialization")
        raise OSError("synthetic serialization failure")

    def fail_link(*args):
        raise OSError("synthetic cross-filesystem link failure")

    def fail_copy(source, destination):
        destination.write_bytes(b"incomplete copy")
        raise OSError("synthetic copy failure")

    if stage == "serialization":
        monkeypatch.setattr(session_export.pickle, "dump", fail_dump)
    else:
        monkeypatch.setattr(session_export.os, "link", fail_link)
        monkeypatch.setattr(session_export.shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="synthetic"):
        session_export._publish_session_pickles([], outputs)
    assert all(path.read_bytes() == b"previous accepted output" for path in outputs)
    assert not list(tmp_path.glob(".*.tmp"))
