"""Independent interval, censoring and finite-population null expectations."""

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from fpcode.behavior.sequences.organisation import (
    eligible_pairs,
    exclusive_timeline,
    metric_values,
    organisation_arrays,
    organisation_metrics,
    valid_frame_intervals,
)
from fpcode.statistics.chain_nulls import frequency_expectations, simulate_chain_null
from fpcode.workflows.chain_organisation import pool_organisation_measurements


def stream(tokens, *, starts=None, blocks=None, keep=None):
    starts = np.arange(len(tokens)) * 10 if starts is None else np.array(starts)
    events = pd.DataFrame({"chain_id": tokens, "start_frame": starts,
                           "end_frame_exclusive": starts + 10, "selection_score": np.ones(len(tokens))})
    if blocks is None:
        blocks = np.array([[0, max(10, int(starts[-1] + 10))]]) if len(tokens) else np.array([[0, 10]])
    selected, _, _ = exclusive_timeline(events, blocks, fps=10, recording_end_frame=int(blocks[-1, 1]))
    return organisation_arrays(selected, retained_ids=set(tokens) if keep is None else keep, fps=10)


def counts(arrays, gap=0):
    result, runs = organisation_metrics(arrays, gap_seconds=gap, modelled_seconds=10)
    result.update(frequency_expectations(arrays["ids"], arrays["reference"], eligible_pairs(arrays, gap_seconds=gap)))
    return result, runs


def test_singleton_inclusive_runs_and_censored_ends():
    c, runs = counts(stream(list("AAA")))
    assert (c["self_pairs"], c["runs"], c["run_call_sum"]) == (2, 1, 3)
    assert runs.length_calls.tolist() == [3]
    assert runs.left_censored.all() and runs.right_censored.all()
    assert c["observed_exits"] == 0
    assert metric_values(c)["mean_run_calls"] == 3
    c, runs = counts(stream(list("BAAAC")))
    assert runs.length_calls.tolist() == [1, 3, 1]
    assert runs.complete.tolist() == [False, True, False]
    assert c["complete_run_call_sum"] == 3


def test_returns_exclude_self_persistence_and_missing_or_excluded_neighbours():
    c, _ = counts(stream(list("ABA")))
    assert c["self_pairs"] == 0 and c["return2_hits"] == 1
    c, _ = counts(stream(list("AAA")))
    assert c["return2_hits"] == 0
    c, _ = counts(stream(list("AXA"), keep={"A"}))
    assert c["calls"] == 2 and c["pair_opportunities"] == 0
    c, _ = counts(stream(list("AA"), starts=[0, 20], blocks=np.array([[0, 10], [20, 30]])), gap=5)
    assert c["pair_opportunities"] == 0


def test_native_gap_threshold_and_zero_support():
    a = stream(list("AA"), starts=[0, 21])
    assert not eligible_pairs(a, gap_seconds=1).any()
    assert eligible_pairs(a, gap_seconds=1.1).all()
    c, _ = counts(stream([]))
    values = metric_values(c)
    assert np.isnan(values["self_probability"]) and np.isnan(values["mean_run_calls"])
    assert values["initiations_per_min"] == 0
    c, _ = counts(stream(["A"]))
    assert np.isnan(metric_values(c)["self_probability"])
    assert metric_values(c)["mean_run_calls"] == 1


def test_deterministic_overlap_ties_and_timeline_partition():
    events = pd.DataFrame({"chain_id": ["B", "A", "C"], "start_frame": [0, 0, 20],
                           "end_frame_exclusive": [10, 10, 30], "selection_score": [3, 3, 2]})
    selected, decisions, spans = exclusive_timeline(events, [[0, 30], [40, 50]], fps=10, recording_end_frame=60)
    assert selected.chain_id.tolist() == ["A", "C"]
    assert decisions.rejection_reason.tolist() == ["overlap", "selected", "selected"]
    assert spans.duration_s.sum() == 6
    assert spans.kind.tolist() == ["call", "unmatched", "call", "missing", "unmatched", "missing"]
    intervals = valid_frame_intervals([0, 1, 2, 4, 5, 6], [1, 1, -1, 1, -1, 1])
    assert intervals.tolist() == [[0, 2], [4, 5], [6, 7]]


def test_analytic_null_matches_enumeration_with_fixed_sparse_slots():
    ids = np.array([0, 0, 0, 1, 1])
    pairs = np.array([True, True, False, True])
    expected = frequency_expectations(ids, np.zeros(5, dtype=int), pairs)
    self_counts, returns = [], []
    for sequence in set(permutations(ids)):
        sequence = np.array(sequence)
        self_counts.append(np.sum(pairs & (sequence[:-1] == sequence[1:])))
        returns.append(int(sequence[0] == sequence[2] and sequence[1] != sequence[0]))
    assert expected["frequency_expected_self_count"] == pytest.approx(np.mean(self_counts))
    assert expected["frequency_expected_return2_count"] == pytest.approx(np.mean(returns))
    with pytest.raises(ValueError):
        frequency_expectations([0, 0], [0, 1], [True])


def test_null_precision_reference_barriers_and_positive_persistence():
    arrays = stream(list("AAAABBBB"))
    c, _ = counts(arrays)
    observed = metric_values(c)
    assert observed["self_excess"] > 0
    null = simulate_chain_null(arrays, pairs=eligible_pairs(arrays, gap_seconds=0), kind="frequency", observed=observed,
                               seed=4, replicates=1999, memory_mb=1).set_index("metric")
    assert null.loc["self_probability", "null_mean"] == pytest.approx(3 / 7, abs=.02)
    assert null.loc["self_probability", "replicates"] == 1999
    assert (null.conservative_batch_bytes <= 1024 ** 2).all()
    markov = simulate_chain_null(arrays, pairs=eligible_pairs(arrays, gap_seconds=0), kind="first_order", observed=observed,
                                 seed=4, replicates=1999, memory_mb=1)
    assert "self_probability" not in set(markov.metric)
    barrier = stream(list("AAXBB"), keep={"A", "B"})
    c, _ = counts(barrier)
    assert metric_values(c)["self_excess"] == 0
    null = simulate_chain_null(barrier, pairs=eligible_pairs(barrier, gap_seconds=0), kind="frequency", observed=metric_values(c),
                               seed=4, replicates=99, memory_mb=1)
    assert null.set_index("metric").loc["self_probability", "null_mean"] == 1


def test_recording_pool_uses_opportunity_weight_and_preserves_mouse_count():
    first, _ = counts(stream(list("AAA")))
    second, _ = counts(stream(list("AB")))
    table = pd.DataFrame([{**c, **metric_values(c), "animal_id": "mouse", "condition": "wave"} for c in (first, second)])
    pooled = pool_organisation_measurements(table, identity=["animal_id", "condition"])
    assert len(pooled) == 1 and pooled.n_recordings.iloc[0] == 2
    assert pooled.self_probability.iloc[0] == pytest.approx(2 / 3)
    assert pooled.mean_run_calls.iloc[0] == pytest.approx(5 / 3)


def test_parallel_mouse_contrasts_preserve_results_and_global_corrections():
    from fpcode.statistics.behavior_comparisons import compare_feature_atlas, compare_feature_atlas_parallel

    table = pd.DataFrame([{"session_id": f"{animal}_{wave}", "animal_id": str(animal),
                           "group": "first" if animal < 4 else "second", "condition": wave,
                           "family": "family", "feature": feature,
                           "value": np.sin(animal + phase) + (wave == "post") * animal / 7}
                          for animal in range(8) for wave in ("pre", "post")
                          for feature, phase in (("a", 0), ("b", 1))])
    settings = {"groups": ("first", "second"), "untreated": ("pre",), "drug_condition": "post",
                "baseline_condition": "pre", "resamples": 99}
    serial = compare_feature_atlas(table, **settings)
    parallel = compare_feature_atlas_parallel(table, workers=2, chunk_features=1, memory_mb=1, **settings)
    for name in serial:
        order = [c for c in ("family", "feature", "view", "baseline", "target", "group", "animal_id", "condition")
                 if c in serial[name]]
        pd.testing.assert_frame_equal(serial[name].sort_values(order).reset_index(drop=True),
                                      parallel[name].sort_values(order).reset_index(drop=True))
