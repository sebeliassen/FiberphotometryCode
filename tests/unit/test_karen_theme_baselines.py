"""Primitive baselines conserve exposure and preserve raw/native barriers."""

import numpy as np
import pandas as pd
import pytest

from studies.karen_pff_vs_monomer.analysis.theme_baselines import _consume, _initial, _rows


def _toy():
    # Raw0/1 both grouped0; unknown raw2 remains unresolved. A different block
    # prevents merging the last two equal grouped states despite native contact.
    return pd.DataFrame({"session_id": ["s"] * 6, "token": [0, 1, 2, 0, 1, 1],
                         "start_frame": [0, 3, 5, 6, 9, 11], "end_frame_exclusive": [3, 5, 6, 9, 11, 13],
                         "block": [0, 0, 0, 0, 0, 1],
                         "mean_speed_px_s": [10, 20, np.nan, 10, 20, 30],
                         "mean_abs_angular_velocity_rad_s": [1, 2, np.nan, 1, 2, 3]})


def _summarise(chunk_size=100):
    states = {"s": _initial()}
    toy = _toy()
    for start in range(0, len(toy), chunk_size):
        _consume(toy.iloc[start:start + chunk_size], states, np.array([0, 0, 20]))
    sessions = pd.DataFrame([dict(session_id="s", animal_id="mouse", group="PFF", wave="3w_PI", fps=10)])
    return states, _rows(states, sessions, {"s": 1.3})


def test_primitive_merged_bouts_units_conservation_and_chunk_boundaries():
    states, rows = _summarise()
    _, chunked = _summarise(chunk_size=1)
    pd.testing.assert_frame_equal(rows, chunked)
    t0 = rows[rows.theme == "__primitive_token0__"].set_index("feature")
    assert t0.loc["bouts_per_min", "numerator"] == 3
    assert t0.loc["bouts_per_min", "value"] == pytest.approx(3 * 60 / 1.3)
    fractions = rows[rows.feature == "time_fraction"]
    assert fractions.value.sum() == pytest.approx(1)
    assert fractions.query("theme == '__unresolved_annotation__'").value.iloc[0] == pytest.approx(1 / 13)
    speed = rows.query("feature == 'within_raw_bout_mean_speed_px_s'").iloc[0]
    assert speed.denominator == 7  # (3-1)+(2-1)+0+(3-1)+(2-1)+(2-1)
    assert speed.value == pytest.approx(110 / 7)
    angular = rows.query("feature == 'within_raw_bout_mean_abs_angular_velocity_rad_s'").iloc[0]
    assert angular.value == pytest.approx(11 / 7)
    assert states["s"]["raw_bouts"] == 6


def test_exposure_mismatch_and_reordering_fail_loudly():
    states, _ = _summarise()
    sessions = pd.DataFrame([dict(session_id="s", animal_id="m", group="PFF", wave="3w_PI", fps=10)])
    with pytest.raises(ValueError, match="conserve"):
        _rows(states, sessions, {"s": 2})
    with pytest.raises(ValueError, match="reordered"):
        _consume(_toy().iloc[:1], states, np.array([0, 0, 20]))
