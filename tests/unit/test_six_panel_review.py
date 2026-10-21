"""Layout and selection invariants for six-execution sampling-mode movies."""

import numpy as np
import pytest

from studies.sequence_viz.render_generator_review_grids import grid_filters, select_movie_sets


def test_six_panel_layout_has_two_rows_three_columns_and_no_time_stretching():
    text = grid_filters([40, 50, 60, 70, 80, 90], columns=3)
    assert "layout=0_0|720_0|1440_0|0_960|720_960|1440_960" in text
    assert "xstack=inputs=6" in text
    assert "tpad=stop=50:stop_mode=add" in text
    assert "tpad=stop=0:stop_mode=add" in text
    assert "setpts=PTS-STARTPTS" in text
    assert "setpts=2" not in text and "stop_mode=clone" not in text


def test_original_three_panel_layout_still_uses_one_row():
    text = grid_filters([30, 60, 40])
    assert "hstack=inputs=3" in text and "xstack" not in text


@pytest.mark.parametrize("counts,columns", [([], 3), ([10, 0, 10], 3), ([10] * 5, 3)])
def test_layout_rejects_missing_empty_or_incomplete_panels(counts, columns):
    with pytest.raises(ValueError):
        grid_filters(counts, columns=columns)


def test_six_panel_modes_preserve_p80_first_three_and_uniform_random_selection():
    calls = [dict(recording=f"r{i}", subject=str(i % 20), start_frame=i) for i in range(200)]
    distances = np.arange(200, dtype=float)
    percentiles = 100 * (199 - distances) / 199
    original = select_movie_sets(calls, distances, percentiles, 17, 0, panels=3)
    expanded = select_movie_sets(calls, distances, percentiles, 17, 0, panels=6)
    assert [e["recording"] for e in expanded["p80"][:3]] == [e["recording"] for e in original["p80"]]
    for mode, examples in expanded.items():
        assert len(examples) == 6 and len({e["recording"] for e in examples}) == 6
        assert all(e["selection_mode"] == mode for e in examples)
    assert all(abs(e["actual_percentile"] - 80) <= 5 for e in expanded["p80"])
    assert expanded["p100"][0]["recording"] == "r0"
    expected = np.random.default_rng(17).choice(200, size=6, replace=False)
    assert [e["recording"] for e in expanded["random"]] == [f"r{i}" for i in expected]
