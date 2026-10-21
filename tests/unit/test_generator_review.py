"""Native interval estimands used to suggest qualitative generator comparisons."""

import pytest

from studies.karen_pff_vs_monomer.analysis.build_core_review import (
    interval_intersection,
    merge_intervals,
)


def test_native_union_counts_nested_and_duplicate_calls_once():
    union = merge_intervals([(8, 12), (2, 6), (3, 5), (2, 6), (5, 9), (20, 21)])
    assert union == [(2, 12), (20, 21)]
    assert sum(stop - start for start, stop in union) == 11


def test_half_open_endpoints_do_not_share_a_frame():
    assert interval_intersection([(5, 10)], [(10, 15)]) == 0
    assert interval_intersection([(5, 10)], [(9, 15)]) == 1


def test_union_intersection_preserves_disjoint_gaps():
    first = merge_intervals([(0, 3), (7, 11)])
    second = merge_intervals([(2, 8), (10, 15)])
    assert interval_intersection(first, second) == 3
    assert interval_intersection(second, first) == 3
    assert interval_intersection([], second) == 0


@pytest.mark.parametrize("interval", [(8, 8), (9, 4)])
def test_invalid_intervals_are_not_silently_repaired(interval):
    with pytest.raises(ValueError, match="invalid native interval"):
        merge_intervals([interval])
