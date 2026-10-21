"""Diagnostic expectations independent of any episode fitter or phenotype."""

from dataclasses import replace

import numpy as np
import pytest

from fpcode.behavior.analysis.unit_views import aggregate_unit_composition, measure_unit_composition
from fpcode.contracts import FrameSpan, SessionKey, UnitOccurrence, UnitStream, UnitType


def _pair(sequences, types, *, subject="mouse", recording="recording", source_model="kpms"):
    session = SessionKey(recording, subject, "study")
    source_events, calls = [], []
    cursor = 0
    for i, (sequence, type_id) in enumerate(zip(sequences, types)):
        start = cursor
        for j, letter in enumerate(sequence):
            source_events.append(UnitOccurrence(f"p{i}-{j}", letter, FrameSpan(cursor, cursor+2)))
            cursor += 2
        calls.append(UnitOccurrence(f"e{i}", type_id, FrameSpan(start, cursor)))
    metadata = dict(session=session, clock="video", fps=10., valid_intervals=(FrameSpan(0, cursor),))
    source = UnitStream(level="primitive", model_id=source_model,
                        catalog=tuple(UnitType(t) for t in "ABCDX"), occurrences=tuple(source_events), **metadata)
    stream = UnitStream(level="episode", model_id="episodes", catalog=tuple(UnitType(t) for t in sorted(set(types))),
                        occurrences=tuple(calls), **metadata)
    return stream, source


def test_diverse_composition_and_exact_loops_remain_separate_descriptors():
    stream, source = _pair(["ABAB", "ABCD", "AAA"], ["loop", "routine", "single_action"])
    table = measure_unit_composition(stream, source)
    assert table.distinct_lower_types.tolist() == [2, 4, 1]
    assert table.shortest_repeat_period_events.tolist() == [2, 4, 1]
    assert table.exact_repeat_copies.tolist() == [2, 1, 3]
    np.testing.assert_allclose(table.event_entropy_bits, [1., 2., 0.])
    summary = aggregate_unit_composition([table])
    assert set(summary) == {"types", "catalogues"}
    assert summary["catalogues"].iloc[0].singleton_types == 3
    assert "quality_score" not in summary["types"]


def test_similar_executions_group_without_exact_matching_and_animals_pool():
    first = measure_unit_composition(*_pair(["ABC", "ABXC", "DDDD"], ["routine", "routine", "other"]))
    second = measure_unit_composition(*_pair(["ABC", "DDDD"], ["routine", "other"],
                                           subject="second_mouse", recording="second_recording"))
    metrics = aggregate_unit_composition([first, second])["types"].set_index("type_id")
    assert metrics.loc["routine", "known_animals"] == 2
    assert metrics.loc["routine", "exact_sequence_variants"] == 2
    assert metrics.loc["routine", "within_pairs"] == 3
    assert metrics.loc["routine", "mean_within_sequence_agreement"] == pytest.approx(5/6)
    assert metrics.loc["routine", "mean_other_type_sequence_agreement"] == 0.


def test_partial_edges_missing_support_and_declared_boundaries_stay_visible():
    stream, source = _pair(["ABCD"], ["routine"])
    stream = replace(stream, occurrences=(UnitOccurrence("clipped", "routine", FrameSpan(1, 7)),
                                         UnitOccurrence("all", "routine", FrameSpan(0, 8))))
    complete = measure_unit_composition(stream, source)
    assert complete.lower_support_complete.tolist() == [True, True]
    assert complete.edge_bout_clipped.tolist() == [True, False]
    metrics = aggregate_unit_composition([complete])["types"].iloc[0]
    assert metrics.comparable_occurrences == 1 and metrics.clipped_edge_occurrences == 1
    source = replace(source, occurrences=source.occurrences[:1]+source.occurrences[2:])
    partial = measure_unit_composition(stream, source)
    assert not partial.lower_support_complete.any()
    assert partial.iloc[1].lower_support_fraction == .75
    source = replace(source, valid_intervals=(FrameSpan(0, 4), FrameSpan(4, 8)))
    assert not measure_unit_composition(stream, source).within_one_lower_valid_block.any()


def test_catalogue_identities_cannot_pool_foreign_vocabularies_or_duplicates():
    pair = _pair(["ABC", "DDD"], ["routine", "other"])
    table = measure_unit_composition(*pair)
    foreign = measure_unit_composition(*_pair(["ABC", "DDD"], ["routine", "other"], source_model="different_kpms"))
    assert len(aggregate_unit_composition([table, foreign])["catalogues"]) == 2
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_unit_composition([table, table])
    changed_catalogue = measure_unit_composition(pair[0], replace(pair[1], catalog=pair[1].catalog+(UnitType("new"),)))
    with pytest.raises(ValueError, match="identical catalogues"):
        aggregate_unit_composition([table, changed_catalogue])
    different_rate = measure_unit_composition(replace(pair[0], fps=20., session=SessionKey("other", "other", "study")),
                                             replace(pair[1], fps=20., session=SessionKey("other", "other", "study")))
    with pytest.raises(ValueError, match="clock/fps"):
        aggregate_unit_composition([table, different_rate])
    with pytest.raises(ValueError, match="clock"):
        measure_unit_composition(pair[0], replace(pair[1], clock="different_clock"))
    with pytest.raises(ValueError, match="overlapping"):
        measure_unit_composition(pair[0], replace(pair[1], occurrences=pair[1].occurrences+
                                                (UnitOccurrence("overlap", "A", FrameSpan(0, 1)),)))


def test_pair_and_sequence_budgets_are_recorded_and_reproducible():
    pair = _pair(["ABC"]*40+["DDD"]*40, ["routine"]*40+["other"]*40)
    table = measure_unit_composition(*pair)
    first = aggregate_unit_composition([table], max_pairs_per_type=7, seed=3)
    second = aggregate_unit_composition([table], max_pairs_per_type=7, seed=3)
    assert first["types"].equals(second["types"])
    assert first["types"].within_pairs.tolist() == [7, 7]
    assert first["types"].other_type_pairs.tolist() == [7, 7]
    capped = measure_unit_composition(*pair, max_sequence_events=2)
    assert capped.sequence_over_budget.all()
    assert capped.lower_events.tolist() == [3]*80
    assert capped.source_sequence.isna().all()
    assert aggregate_unit_composition([capped])["types"].comparable_occurrences.tolist() == [0, 0]


def test_unknown_animals_and_empty_catalogues_are_honest():
    stream, source = _pair(["ABC"], ["routine"], subject=None)
    result = aggregate_unit_composition([measure_unit_composition(stream, source)])
    assert result["types"].iloc[0].known_animals == 0
    assert result["types"].iloc[0].unknown_subject_occurrences == 1
    stream = replace(stream, occurrences=())
    result = aggregate_unit_composition([measure_unit_composition(stream, source)])
    assert result["catalogues"].iloc[0].occurrences == 0
    assert result["types"].iloc[0].occurrences == 0
    assert np.isnan(result["types"].iloc[0].mean_within_sequence_agreement)
