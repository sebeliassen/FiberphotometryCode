"""Saved metadata auditing preserves original identity, missingness and clocks."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from studies.session_model_exports.inspect_behavior_inputs import (
    export_label_tables,
    inspect_session,
    main,
)


def test_original_drug_metadata_suffixes_and_missing_signal_support():
    table = pd.DataFrame({"syllable": pd.array([58, None, 60], dtype="Int64"),
                          "signal_2_zscore_datta": [np.nan, 1., np.inf]},
                         index=[100, 101, 105])
    session = SimpleNamespace(session_id="recording", metadata={"drug": "Saline",
                              "injection_frame": 102, "signal_meta": {"signal_2": {"brain_reg": "DMS"}}},
                              all_data_df=table)
    result = inspect_session(session)
    assert result["metadata"]["drug"] == "Saline"
    assert result["metadata"]["injection_frame"] == 102
    desc = result["tables"]["all_data_df"]
    assert desc["labels"] == {"count": 2, "types": [58, 60]}
    assert desc["signals"]["signal_2_zscore_datta"]["finite_rows"] == 1
    assert desc["signals"]["signal_2_zscore_datta"]["first_finite_index"] == 101
    assert table.syllable.isna().tolist() == [False, True, False]


def test_export_keeps_sparse_original_indices_and_labels(tmp_path):
    session = SimpleNamespace(session_id="a", syllables=pd.DataFrame({
        "syllable": pd.array([58, None, 60], dtype="Int64")}, index=[49922, 49924, 50000]))
    paths = export_label_tables(session, tmp_path)
    with np.load(paths["syllables"], allow_pickle=False) as result:
        np.testing.assert_array_equal(result["table_index"], [49922, 50000])
        np.testing.assert_array_equal(result["labels"], [58, 60])


def test_dry_run_does_not_decode_or_create_output(tmp_path, capsys):
    source = tmp_path / "input.pkl"
    source.write_bytes(b"not a pickle")
    output = tmp_path / "audit"
    assert main(["--input", str(source), "--output", str(output)]) == 0
    assert not output.exists()
    assert '"execute": false' in capsys.readouterr().out
