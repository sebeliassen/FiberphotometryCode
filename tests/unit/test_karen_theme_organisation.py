"""Independent clock, barrier, overlap and finite-population theme examples."""

import itertools

import numpy as np
import pandas as pd
import pytest

from studies.karen_pff_vs_monomer.analysis.theme_organisation import session_readouts


def _calls(themes, starts=None, durations=None, scores=None):
    starts = np.arange(len(themes)) * 10 if starts is None else np.asarray(starts)
    durations = np.full(len(themes), 10) if durations is None else np.asarray(durations)
    return pd.DataFrame({"occurrence_id": [f"call-{i}" for i in range(len(themes))],
                         "theme": themes, "start_frame": starts,
                         "end_frame_exclusive": starts + durations,
                         "score": np.ones(len(themes)) if scores is None else scores})


def _row(result, feature, theme="__catalogue__", gap=1):
    table = result["readouts"]
    which = (table.feature == feature) & (table.theme == theme)
    which &= table.gap_seconds.isna() if gap is None else table.gap_seconds == gap
    assert which.sum() == 1
    return table.loc[which].iloc[0]


def test_exact_sequence_rates_transitions_and_returns():
    # A A B A C A: 1/5 repeats, A->A 1/3, A->B 1/3, A->C 1/3,
    # B->A and C->A 1/1; two nontrivial A-X-A returns in four slots.
    r = session_readouts(_calls(list("AABACA")), [[0, 60]], fps=10)
    assert _row(r, "self_repeat_probability").value == pytest.approx(1 / 5)
    assert _row(r, "self_repeat_probability", "A").value == pytest.approx(1 / 3)
    assert _row(r, "transition_to:B", "A").value == pytest.approx(1 / 3)
    assert _row(r, "transition_to:A", "B").value == 1
    assert _row(r, "return2_probability").value == .5
    assert _row(r, "return2_probability", "A").value == pytest.approx(2 / 3)
    assert _row(r, "mean_run_calls").value == pytest.approx(6 / 5)
    assert _row(r, "observed_entries_per_min").value == 40
    assert _row(r, "exclusive_calls_per_min", gap=None).value == 60
    assert _row(r, "exclusive_coverage_fraction", gap=None).value == 1
    assert _row(r, "within_valid_block_recurrence_gap_s", "A", gap=None).value == pytest.approx(2 / 3)
    assert len(r["timeline"]) == 6


def test_overlap_priority_and_tie_reorder_are_reproducible():
    calls = _calls(["B", "A", "C", "A"], starts=[0, 0, 20, 0], durations=[20, 20, 10, 20], scores=[.5, .9, .4, .9])
    # lexical theme A wins B; same-theme complete tie resolves lexical occurrence.
    left = session_readouts(calls, [[0, 40]], fps=10)
    right = session_readouts(calls.iloc[::-1], [[0, 40]], fps=10)
    assert list(left["selected"].occurrence_id) == ["call-1", "call-2"]
    assert list(right["selected"].occurrence_id) == ["call-1", "call-2"]
    assert (left["decisions"].rejection_reason == "overlap").sum() == 2
    assert _row(left, "exclusive_coverage_fraction", gap=None).value == .75
    assert left["timeline"].iloc[-1].kind == "unmatched"


def test_missing_native_frames_and_excluded_themes_are_barriers():
    # X is selected then excluded; removing it first would fabricate A->A.
    calls = _calls(["A", "X", "A", "B", "A"], starts=[0, 10, 20, 40, 50])
    r = session_readouts(calls, [[0, 30], [40, 60]], fps=10, theme_ids=["A", "B", "absent"])
    assert _row(r, "self_repeat_probability").denominator == 1
    assert _row(r, "self_repeat_probability").value == 0
    assert _row(r, "return2_probability").denominator == 0
    assert np.isnan(_row(r, "return2_probability").value)
    assert r["decisions"].query("theme == 'X'").iloc[0].analysis_reason == "excluded_theme"
    assert set(r["pairs"].g1_reason) == {"excluded_call", "missing_boundary", "eligible"}
    assert _row(r, "exclusive_calls_per_min", "absent", gap=None).value == 0
    assert np.isnan(_row(r, "self_repeat_probability", "absent").value)
    assert np.isnan(_row(r, "within_valid_block_recurrence_gap_s", "A", gap=None).value)
    assert r["timeline"].query("kind == 'missing'").duration_s.sum() == 1


def test_exact_gap_sensitivity_does_not_treat_unmatched_time_as_missing():
    calls = _calls(["A", "A", "B"], starts=[0, 20, 45])
    r = session_readouts(calls, [[0, 60]], fps=10)
    assert _row(r, "self_repeat_probability", gap=0).denominator == 0
    assert _row(r, "self_repeat_probability", gap=1).value == 1
    assert _row(r, "self_repeat_probability", gap=2).value == .5
    assert r["timeline"].query("kind == 'unmatched'").duration_s.sum() == 3
    assert r["summary"]["valid_seconds"] == 6


def test_frequency_expectations_match_exhaustive_permutations():
    themes = list("AABC")
    # Fixed gaps remove one pair slot, without changing frequency reference block.
    r = session_readouts(_calls(themes, starts=[0, 10, 30, 40]), [[0, 50]], fps=10, gap_seconds=(0,))
    sequences = set(itertools.permutations(themes))
    self_counts, return_counts = [], []
    for x in sequences:
        self_counts.append(int(x[0] == x[1]) + int(x[2] == x[3]))
        # No two eligible neighbouring pair slots, hence no return opportunity.
        return_counts.append(0)
    expected = _row(r, "frequency_expected_self_probability", gap=0)
    assert expected.numerator == pytest.approx(np.mean(self_counts))
    assert expected.denominator == 2
    assert np.isnan(_row(r, "frequency_expected_return2_probability", gap=0).value)
    # All eligible: ABA chance enumerated independently, excluding AAA.
    r = session_readouts(_calls(themes), [[0, 40]], fps=10, gap_seconds=(0,))
    returns = [sum(x[i] == x[i + 2] and x[i + 1] != x[i] for i in range(2)) for x in sequences]
    assert _row(r, "frequency_expected_return2_probability", gap=0).numerator == pytest.approx(np.mean(returns))


def test_empty_and_singleton_opportunities_stay_missing():
    r = session_readouts([], [[0, 60]], theme_ids=["A"])
    assert _row(r, "exclusive_calls_per_min", "A", gap=None).value == 0
    assert np.isnan(_row(r, "self_repeat_probability").value)
    assert len(r["decisions"]) == 0
    r = session_readouts(_calls(["A"]), [[0, 10]], fps=10)
    assert _row(r, "mean_run_calls").value == 1
    assert _row(r, "observed_entries_per_min").value == 0
    assert _row(r, "right_censored_run_fraction").value == 1
    assert np.isnan(_row(r, "self_repeat_probability").value)


def test_invalid_intervals_and_identities_are_not_silently_coerced():
    calls = _calls(["A"])
    calls["start_frame"] = calls.start_frame.astype(float)
    calls.loc[0, "start_frame"] = .5
    with pytest.raises(ValueError, match="integer frames"):
        session_readouts(calls, [[0, 10]])
    with pytest.raises(ValueError, match="ordered, disjoint"):
        session_readouts(_calls(["A"]), [[0, 20], [10, 30]])
    calls = _calls(["A", "B"])
    calls["occurrence_id"] = "duplicate"
    with pytest.raises(ValueError, match="unique"):
        session_readouts(calls, [[0, 20]])


def test_outside_valid_block_call_is_audited_and_cannot_bridge_gap():
    calls = _calls(["A", "B", "A"], starts=[0, 9, 20], durations=[10, 12, 10])
    r = session_readouts(calls, [[0, 10], [20, 30]], fps=10)
    assert len(r["selected"]) == 2
    assert r["decisions"].query("theme == 'B'").iloc[0].rejection_reason == "outside_valid_block"
    assert _row(r, "self_repeat_probability").denominator == 0
    assert _row(r, "observed_entries_per_min").value == 0


def test_optional_first_order_null_is_seeded_bounded_and_does_not_test_self():
    from studies.karen_pff_vs_monomer.analysis.theme_organisation import first_order_readouts

    # A-B-C-A-B-C deterministic eligible transitions imply every lag3 slot returns.
    result = session_readouts(_calls(list("ABCABCABC")), [[0, 90]], fps=10, gap_seconds=(0,))
    null = first_order_readouts(result, replicates=199, seed=8, memory_mb=1, gap_seconds=0)
    assert set(null["null_summary"].metric) == {"return2_probability", "return3_probability"}
    assert (null["null_summary"].replicates == 199).all()
    assert (null["readouts"].value == 0).all()
    assert (null["readouts"].p_surrogate_two_sided == 1).all()
    duplicate = first_order_readouts(result, replicates=199, seed=8, memory_mb=1, gap_seconds=0)
    pd.testing.assert_frame_equal(null["null_summary"], duplicate["null_summary"])
    with pytest.raises(MemoryError):
        first_order_readouts(result, replicates=199, memory_mb=.0001)


def test_context_requires_support_both_sides_and_keeps_execution_units():
    from studies.karen_pff_vs_monomer.analysis.theme_organisation import context_readouts

    themes = list("AABAABAABA")
    calls = _calls(themes)
    # A after A speed=10; A after B speed=2. Other executions are irrelevant.
    calls["speed"] = [99, 10, 0, 2, 10, 0, 2, 10, 0, 2]
    table = context_readouts(calls, [[0, 100]], fps=10, gap_seconds=0)
    a = table[(table.theme == "A") & (table.feature == "preceding_same_minus_other:speed")].iloc[0]
    assert a.n_same == 3 and a.n_other == 3
    assert a.value == 8 and a.status == "supported"
    duration = table[(table.theme == "A") & (table.feature == "preceding_same_minus_other:duration_s")].iloc[0]
    assert duration.value == 0
    b = table[(table.theme == "B") & (table.feature == "preceding_same_minus_other:speed")].iloc[0]
    assert b.n_same == 0 and np.isnan(b.value) and b.status == "insufficient_support"


def test_context_cannot_bridge_gap_or_exclusion_and_logs_missing_metric():
    from studies.karen_pff_vs_monomer.analysis.theme_organisation import context_readouts

    calls = _calls(list("AABAABAABA"))
    calls["speed"] = 1.
    # Break one formerly A->A context at a native missing frame; no 3+3 support.
    calls.loc[0, "end_frame_exclusive"] = 9
    table = context_readouts(calls, [[0, 9], [10, 100]], fps=10)
    a = table[(table.theme == "A") & (table.feature == "preceding_same_minus_other:speed")].iloc[0]
    assert a.n_same == 2 and np.isnan(a.value)
    calls["included"] = calls.theme != "B"
    table = context_readouts(calls, [[0, 9], [10, 100]], fps=10, metrics=("duration_s", "unknown_metric"))
    a = table[(table.theme == "A") & (table.feature == "preceding_same_minus_other:duration_s")].iloc[0]
    assert a.n_other == 0 and np.isnan(a.value)
    assert set(table.loc[table.feature == "preceding_same_minus_other:unknown_metric", "status"]) == {"missing_metric"}
    overlapping = _calls(["A", "B"], starts=[0, 5])
    with pytest.raises(ValueError, match="already be exclusive"):
        context_readouts(overlapping, [[0, 20]])
