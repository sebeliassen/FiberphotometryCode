"""Independent preservation and split/shift expectations for capacity diagnostics."""

import numpy as np
import pandas as pd
import pytest

from studies.karen_pff_vs_monomer.analysis.reference_capacity import (
    eulerian_shuffle,
    execution_masks,
    execution_overlap,
    order_null,
    template_cosines,
    transition_counts,
)


def test_exact_transition_null_preserves_counts_and_endpoints_without_self_edges():
    z = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2, 1, 0])
    original = transition_counts(z, 3)
    variants = set()
    for seed in range(10):
        actual = eulerian_shuffle(z, np.random.RandomState(seed))
        np.testing.assert_array_equal(transition_counts(actual, 3), original)
        np.testing.assert_array_equal(np.bincount(actual), np.bincount(z))
        assert actual[0] == z[0] and actual[-1] == z[-1]
        assert np.all(actual[1:] != actual[:-1])
        variants.add(tuple(actual))
    assert len(variants) > 1


@pytest.mark.parametrize("kind", ["frequency", "first_order"])
def test_nulls_keep_padding_and_per_block_counts(kind):
    z = np.array([0, 1, 2, 0, -1, -1, -1, 2, 1, 2, 1, -1, -1, -1])
    result, audit = order_null(z, kind, 45, 3)
    assert np.array_equal(result[z < 0], z[z < 0])
    for a, b in ((0, 4), (7, 11)):
        np.testing.assert_array_equal(np.bincount(z[a:b], minlength=3), np.bincount(result[a:b], minlength=3))
    assert audit["padding_preserved"]


def test_physical_overlap_counts_union_once_and_exposes_split_containment():
    parent = np.array([[1, 1, 1, 1, 0]], bool)
    child = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0]], bool)
    jac, in_parent, in_child = execution_overlap(parent, child, np.array([1.0, 2.0, 3.0, 4.0, 10.0]))
    np.testing.assert_allclose(jac, [[0.3, 0.7]])
    np.testing.assert_allclose(in_parent, [[0.3, 0.7]])
    np.testing.assert_allclose(in_child, [[1.0, 1.0]])


def test_partial_shift_matching_cannot_claim_perfect_similarity_from_one_shared_stage():
    a = np.eye(3, dtype=np.float32)[:, None, :]
    b = np.roll(a, 1, axis=2)
    locked, shifted = template_cosines(a, b)
    assert locked[0, 0] == pytest.approx(0.0)
    assert shifted[0, 0] == pytest.approx(2 / 3)
    same, _ = template_cosines(a, a)
    assert same[0, 0] == pytest.approx(1.0)


def test_execution_masks_union_overlapping_calls():
    calls = pd.DataFrame({"factor": [0, 0, 1], "start": [1, 2, 0], "end_exclusive": [4, 5, 3]})
    actual = execution_masks(calls, 2, 7)
    assert actual[0].tolist() == [False, True, True, True, True, False, False]
    assert actual[1].sum() == 3
