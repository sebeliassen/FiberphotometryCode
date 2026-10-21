"""Independent geometry, animal support and native overlap selection checks."""

import json

import numpy as np
import pytest

from studies.karen_pff_vs_monomer.analysis.theme_neighbourhoods import select_neighbourhoods


def calls(n, mice=6):
    return [dict(recording=f"recording_{i}", subject=f"mouse_{i % mice}",
                 start_frame=10, stop_frame=20, occurrence_id=f"occurrence_{i:03}") for i in range(n)]


def test_three_supported_separated_forms_represent_three_actual_mice_each():
    values = np.concatenate([np.arange(6) * .01 + center for center in (0., 10., 20.)])[:, None]
    records = calls(len(values))
    result = select_neighbourhoods(values, records)
    assert len(result["regions"]) == 3 and len(set(result["selected_indices"])) == 9
    represented = set()
    for region in result["regions"]:
        selected = region["selected_indices"]
        represented.add(selected[0] // 6)
        assert len({index // 6 for index in selected}) == 1
        assert region["selected_mice"] == 3
        assert region["support_mice"] == 6
        assert region["nearest_partition_count"] == 6
        assert region["support_count"] == 6
        for index, squared in zip(selected, region["selected_distances_squared"]):
            assert squared == pytest.approx(float((values[index] - values[selected[0]])[0] ** 2))
    assert represented == {0, 1, 2}
    assert result["regions"][1]["status"] == result["regions"][2]["status"] == "separated_local_neighbourhood"
    assert sum(row["nearest_partition_count"] for row in result["regions"]) == len(values)
    json.dumps(result, allow_nan=False)


def test_many_repeats_from_one_mouse_do_not_supply_cross_mouse_density():
    # Prolific isolated mouse contributes 30 identical executions, but all nearby
    # cross-mouse support is at 20. A real six-mouse neighbourhood is near zero.
    values = np.r_[np.linspace(0., .05, 6), np.repeat(20., 30)][:, None]
    records = calls(len(values))
    for row in records[6:]:
        row["subject"] = "prolific_mouse"
    result = select_neighbourhoods(values, records, regions=1)
    region = result["regions"][0]
    assert region["anchor_index"] < 6
    assert region["support_count"] == 6 and region["support_mice"] == 6
    assert result["n_mice"] == 7
    assert region["support_animal_balanced_share"] == pytest.approx(6 / 7)


def test_core_overlap_suppression_is_global_but_touching_endpoints_allowed():
    values = np.arange(18, dtype=float)[:, None] * .1
    records = calls(18)
    records[1].update(recording="recording_0", subject=records[0]["subject"], start_frame=11, stop_frame=19)
    records[2].update(recording="recording_0", subject=records[0]["subject"], start_frame=20, stop_frame=30)
    result = select_neighbourhoods(values, records)
    selected = result["selected_indices"]
    for a in selected:
        for b in selected:
            if a != b and records[a]["recording"] == records[b]["recording"]:
                assert records[a]["stop_frame"] <= records[b]["start_frame"] or records[b]["stop_frame"] <= records[a]["start_frame"]
    adjacent = calls(9)
    adjacent[1].update(recording="recording_0", subject=adjacent[0]["subject"], start_frame=20, stop_frame=30)
    allowed = select_neighbourhoods(np.zeros((9, 1)), adjacent)
    assert set(allowed["selected_indices"]) == set(range(9))


def test_identical_population_is_not_claimed_to_have_three_distinct_clusters():
    result = select_neighbourhoods(np.zeros((12, 449)), calls(12))
    assert all(region["support_count"] == 12 for region in result["regions"])
    assert [region["status"] for region in result["regions"]][1:] == ["overlapping_variation_anchor"] * 2
    assert [region["nearest_partition_count"] for region in result["regions"]] == [12, 0, 0]
    assert result["population_local_radius_overlap"] == result["population_local_radius_coverage"] == 1
    assert all(row["overlapping_local_support"] for row in result["assignments"])


def test_ties_are_identity_deterministic_and_phenotype_is_unused():
    records = calls(12)
    first = select_neighbourhoods(np.zeros((12, 2)), records)
    permutation = np.arange(12)[::-1]
    shuffled = [{**records[index], "group": "PFF" if index % 2 else "Monomer", "wave": "arbitrary"}
                for index in permutation]
    second = select_neighbourhoods(np.zeros((12, 2)), shuffled)
    assert [region["selected_occurrence_ids"] for region in first["regions"]] == [region["selected_occurrence_ids"] for region in second["regions"]]
    assert all("group" not in row and "wave" not in row for row in second["identities"])


def test_partition_distances_match_direct_unchanged_multifeature_geometry():
    values = np.array([[i * 3., (i % 3) * 5.] for i in range(12)])
    frozen = values.copy()
    result = select_neighbourhoods(values, calls(len(values)))
    anchors = [region["anchor_index"] for region in result["regions"]]
    direct = np.sum((values[:, None] - values[anchors][None]) ** 2, axis=2)
    for index, row in enumerate(result["assignments"]):
        nearest = int(np.argmin(direct[index]))
        assert row["nearest_region"] == f"region_{nearest + 1}"
        assert row["distance_squared"] == pytest.approx(direct[index, nearest])
    np.testing.assert_array_equal(values, frozen)


def test_missing_population_support_identity_and_memory_fail_before_large_work():
    with pytest.raises(ValueError, match="three distinct mice"):
        select_neighbourhoods(np.zeros((9, 1)), calls(9, mice=2))
    with pytest.raises(ValueError, match="memory estimate"):
        select_neighbourhoods(np.zeros((9, 449)), calls(9), max_memory_gb=.001)
    records = calls(9)
    records[-1]["occurrence_id"] = records[0]["occurrence_id"]
    with pytest.raises(ValueError, match="unique"):
        select_neighbourhoods(np.zeros((9, 1)), records)
    with pytest.raises(ValueError, match="finite"):
        select_neighbourhoods(np.full((9, 1), np.nan), calls(9))


def test_incompatible_population_returns_fewer_rows_with_explicit_warning():
    records = calls(9, mice=3)
    for index, row in enumerate(records):
        row["recording"] = f"recording_{index % 3}"
    result = select_neighbourhoods(np.zeros((9, 1)), records)
    assert len(result["regions"]) == 1 and len(result["selected_indices"]) == 3
    assert result["warnings"] and result["regions"][0]["selected_mice"] == 3
    assert len(result["assignments"]) == 9
