"""Anonymous, small real-data comparisons; no access to the original recordings.

The fixture contains the first 256 rows of both photometry channels and 512
modeled frame indexes from the first SessionRecord in the existing
syllable_generation/data/sessions_multiview_pilot_mini.pkl. Identifiers,
metadata, tracking and the rest of the sessions are omitted. Reference values
were calculated independently using the historical first-frame-occurrence,
row-position interpolation (four-row limit, both ends) and nearest-endpoint
rules. The two original SessionRecords also loaded through their retained class
namespace during the 2026-10-11 read-only check.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fpcode.adapters.legacy_sessions import as_signal_series
from fpcode.photometry.alignment import combined_photometry_and_syllables
from fpcode.photometry.event_windows import legacy_nearest_window

FIXTURE = Path(__file__).resolve().parents[1] / "golden/historical_session_slice.npz"


def test_historical_photometry_slice_matches_independent_reference():
    with np.load(FIXTURE, allow_pickle=False) as records:
        tables = {}
        for frequency in ("415", "470"):
            table = pd.DataFrame(records[f"values_{frequency}"],
                                 columns=records[f"columns_{frequency}"].tolist())
            table["frame_idx"] = records[f"frame_index_{frequency}"]
            tables[frequency] = table
        sample = SimpleNamespace(photometry=tables, metadata={}, syllables=pd.DataFrame(
            {"syllable": records["syllable"]}, index=records["frame_index"]))
        before = {key: table.copy(deep=True) for key, table in tables.items()}
        expected = pd.DataFrame(records["expected_merged"], index=sample.syllables.index,
                                columns=records["expected_columns"].tolist())
        actual = combined_photometry_and_syllables(sample, freqs=["415", "470"])
        pd.testing.assert_frame_equal(actual, expected)
        for frequency, table in tables.items():
            adapted = as_signal_series(sample, frequency, time_unit="s", clock="FP3002")
            np.testing.assert_array_equal(adapted.time.values, table["SystemTimestamp"].to_numpy())
            np.testing.assert_array_equal(adapted.values, records[f"values_{frequency}"][:, 1:])
            assert not adapted.time.values.flags.writeable and not adapted.values.flags.writeable
            pd.testing.assert_frame_equal(table, before[frequency])


def test_historical_window_slice_matches_saved_nearest_endpoints():
    with np.load(FIXTURE, allow_pickle=False) as records:
        timestamps = records["values_470"][:, 0]
        start, stop = records["window_bounds"]
        assert legacy_nearest_window(timestamps, start, stop) == tuple(records["expected_window"])
