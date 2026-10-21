"""Movie/statistics populations must agree in physical intervals, not ID text."""

import pandas as pd
import pytest

from studies.karen_pff_vs_monomer.analysis.render_theme_showcases import bind_measurements


def population():
    return pd.DataFrame([dict(theme="rear_backoff", session_id="session", start_frame=60,
                              end_frame_exclusive=180, occurrence_id="media:offset100")])


def test_different_bout_offset_ids_bind_to_identical_native_execution():
    calls = population()
    metrics = calls.assign(occurrence_id="measurement:offset96", duration_s=2.)
    bound = bind_measurements(calls, metrics)
    assert bound.loc["media:offset100", "measurement_occurrence_id"] == "measurement:offset96"
    assert bound.loc["media:offset100", "duration_s"] == 2.


def test_identical_id_cannot_hide_changed_native_boundary():
    with pytest.raises(ValueError, match="native occurrence population"):
        bind_measurements(population(), population().assign(end_frame_exclusive=181))


def test_duplicate_and_extra_intervals_cannot_silently_reweight_population():
    calls = population()
    with pytest.raises(ValueError, match="duplicate native"):
        bind_measurements(calls, pd.concat([calls, calls.assign(occurrence_id="duplicate")]))
    extra = calls.assign(start_frame=300, end_frame_exclusive=400, occurrence_id="extra")
    with pytest.raises(ValueError, match="native occurrence population"):
        bind_measurements(calls, pd.concat([calls, extra]))
