"""Real detector representations feed common consumers with explicit semantics."""

import numpy as np
import pandas as pd
import pytest

from fpcode.behavior.analysis.organisation import metric_values
from fpcode.behavior.analysis.unit_views import measure_unit_organisation
from fpcode.behavior.compatibility import StreamRequirements, inspect_compatibility, require_compatible
from fpcode.behavior.episodes.chains import exact_occurrences
from fpcode.behavior.episodes.seqnmf import activation_events
from fpcode.behavior.primitives.resolutions import remap_states
from fpcode.behavior.unit_adapters import (
    from_bout_windows,
    from_frame_labels,
    from_occurrence_table,
    occurrence_table,
    primitive_bout_table,
)
from fpcode.contracts import FrameSpan, SessionKey, UnitOccurrence, UnitStream, UnitType

SESSION = SessionKey("recording", "mouse", "study")


def metadata(catalog, *, model_id="detector", spans=(FrameSpan(0, 30),), level="episode"):
    return dict(session=SESSION, level=level, model_id=model_id, clock="video", fps=10.,
                valid_intervals=spans, catalog=tuple(UnitType(t) for t in catalog))


def test_adjacent_exact_and_seqnmf_outputs_use_same_downstream_analysis():
    primitives = from_frame_labels(np.arange(18), np.repeat([0, 1, 2] * 3, 2),
                                   session=SESSION, model_id="kpms", clock="video", fps=10)
    bouts, crosswalk = primitive_bout_table(primitives)
    assert crosswalk == ("0", "1", "2")
    windows = exact_occurrences(bouts, [0, 1, 2])
    exact = pd.DataFrame([{"chain_id": "exact", "start": a, "end": b} for a, b in windows])
    # Real activation_events uses the full weighted template, without fitting.
    x = np.eye(3, dtype=np.float32)[:, [0, 1, 2] * 3]
    w = np.eye(3, dtype=np.float32)[:, None, :]
    h = np.zeros((1, 9), dtype=np.float32)
    h[0, [0, 3, 6]] = 1
    valid = np.zeros(9, dtype=bool)
    valid[[0, 3, 6]] = True
    factors = pd.DataFrame(activation_events(h, w, x, thresholds=[.5], valid_starts=valid, max_error=.01))
    factors["chain_id"] = "factor"
    streams = [from_bout_windows(exact, bouts, **metadata(["exact"], model_id="exact", spans=primitives.valid_intervals)),
               from_bout_windows(factors, bouts, **metadata(["factor"], model_id="seqnmf", spans=primitives.valid_intervals),
                                 score_column="activation", score_kind="model_score")]
    assert type(streams[0]) is type(streams[1]) is UnitStream
    assert occurrence_table(streams[0]).columns.equals(occurrence_table(streams[1]).columns)
    for stream in streams:
        result = measure_unit_organisation(stream, gap_seconds=0)
        assert result["counts"]["calls"] == 3
        assert result["counts"]["self_pairs"] == 2
        assert result["runs"].length_calls.tolist() == [3]
        assert metric_values(result["counts"])["calls_per_min"] == pytest.approx(100.)
    assert streams[1].occurrences[0].score_kind == "model_score"
    assert streams[0].occurrences[0].score is None


def test_resolution_changes_counts_without_requiring_equal_shapes_or_losing_gaps():
    frames = np.array([0, 1, 2, 3, 6, 7, 8, 9])
    raw = np.array([0, 0, 1, 1, 1, 1, -1, 0])
    unchanged = raw.copy()
    streams = [from_frame_labels(frames, labels, session=SESSION, model_id=name, clock="video", fps=10)
               for name, labels in (("raw", raw), ("coarse", remap_states(raw, [0, 0])))]
    assert len(streams[0].occurrences) == 4 and len(streams[1].occurrences) == 3
    assert streams[0].valid_intervals == streams[1].valid_intervals
    for stream, n in zip(streams, (4, 3)):
        require_compatible(stream, StreamRequirements(levels=("primitive",)))
        result = measure_unit_organisation(stream, gap_seconds=5)["counts"]
        assert result["calls"] == n
        assert result["modelled_seconds"] == .7
        assert result["pair_opportunities"] == (1 if n == 4 else 0)
    np.testing.assert_array_equal(raw, unchanged)


def test_different_table_schema_and_event_count_adapt_without_changing_consumer():
    table = pd.DataFrame({"state": ["a", "b"], "begin": [0, 10], "finish": [10, 30], "belief": [.8, .9]})
    stream = from_occurrence_table(table, **metadata(["a", "b"]), type_column="state", start_column="begin",
                                   stop_column="finish", score_column="belief", score_kind="posterior")
    counts = measure_unit_organisation(stream, gap_seconds=0)["counts"]
    assert counts["calls"] == 2 and counts["self_pairs"] == 0 and counts["pair_opportunities"] == 1
    assert stream.occurrences[1].interval == FrameSpan(10, 30)


def test_overlaps_are_preserved_then_require_explicit_analysis_policy():
    table = pd.DataFrame({"chain_id": ["a", "b"], "start_frame": [0, 5], "end_frame_exclusive": [10, 15], "score": [1., 2.]})
    stream = from_occurrence_table(table, **metadata(["a", "b"]), score_column="score", score_kind="model_score")
    assert len(stream.occurrences) == 2
    with pytest.raises(ValueError, match="overlapping"):
        measure_unit_organisation(stream, gap_seconds=1)
    result = measure_unit_organisation(stream, gap_seconds=1, overlap_policy="score_ranked")
    assert result["counts"]["calls"] == 1
    assert result["runs"].chain_id.tolist() == ["b"]
    assert result["selection_audit"].rejection_reason.tolist() == ["overlap", "selected"]


def test_excluded_unit_remains_barrier_in_common_view():
    table = pd.DataFrame({"chain_id": ["a", "excluded", "a"], "start_frame": [0, 10, 20], "end_frame_exclusive": [10, 20, 30]})
    stream = from_occurrence_table(table, **metadata(["a", "excluded"]))
    result = measure_unit_organisation(stream, gap_seconds=5, retained_type_ids={"a"})
    assert result["counts"]["calls"] == 2 and result["counts"]["pair_opportunities"] == 0
    assert len(result["selection_audit"]) == 3


def test_unsupported_information_and_mismatched_clock_fail_before_consumption():
    stream = UnitStream(occurrences=(), **metadata([]))
    assert inspect_compatibility(stream).compatible
    assert not inspect_compatibility(np.zeros((2, 3))).compatible
    for requirements, text in [(StreamRequirements(required_information=("completion",)), "completion"),
                               (StreamRequirements(clock="photometry"), "clock"),
                               (StreamRequirements(fps=60), "fps"),
                               (StreamRequirements(levels=("primitive",)), "level"),
                               (StreamRequirements(session=SessionKey("other")), "recording")]:
        with pytest.raises(ValueError, match=text):
            require_compatible(stream, requirements)


def test_native_adapters_reject_lossy_coordinates_cross_gap_windows_and_foreign_records():
    table = pd.DataFrame({"chain_id": ["a"], "start_frame": [0.5], "end_frame_exclusive": [10.]})
    with pytest.raises(ValueError, match="integers"):
        from_occurrence_table(table, **metadata(["a"]))
    table.start_frame = 0
    table.end_frame_exclusive = 10
    table["session_id"] = "other"
    with pytest.raises(ValueError, match="recording"):
        from_occurrence_table(table, **metadata(["a"]))
    bouts = pd.DataFrame({"start_frame": [0, 20], "end_frame_exclusive": [10, 30]})
    windows = pd.DataFrame({"chain_id": ["a"], "start": [0], "end": [2]})
    with pytest.raises(ValueError, match="cross"):
        from_bout_windows(windows, bouts, **metadata(["a"]))
    with pytest.raises(ValueError, match="score_kind"):
        from_occurrence_table(pd.DataFrame({"chain_id": ["a"], "start_frame": [0], "end_frame_exclusive": [10], "score": [2.]}),
                              **metadata(["a"]), score_column="score")


def test_empty_and_partial_assignment_use_valid_exposure_without_inventing_events():
    empty = from_occurrence_table(pd.DataFrame(columns=["chain_id", "start_frame", "end_frame_exclusive"]), **metadata(["a"]))
    counts = measure_unit_organisation(empty, gap_seconds=0)["counts"]
    assert counts["calls"] == 0 and counts["modelled_seconds"] == 3
    assert metric_values(counts)["calls_per_min"] == 0
    assert np.isnan(metric_values(counts)["self_probability"])
    partial = UnitStream(occurrences=(UnitOccurrence("one", "a", FrameSpan(0, 5)),
                                     UnitOccurrence("two", "a", FrameSpan(10, 15))), **metadata(["a"], level="primitive"))
    bouts, _ = primitive_bout_table(partial)
    assert bouts.segment.tolist() == [0, 1]
    assert exact_occurrences(bouts, [0, 0]) == []


def test_score_selection_refuses_missing_or_mixed_evidence_kinds():
    events = (UnitOccurrence("a", "a", FrameSpan(0, 10)),)
    with pytest.raises(ValueError, match="scores"):
        measure_unit_organisation(UnitStream(occurrences=events, **metadata(["a"])), gap_seconds=0, overlap_policy="score_ranked")
    events = (UnitOccurrence("a", "a", FrameSpan(0, 10), .8, "posterior"),
              UnitOccurrence("b", "a", FrameSpan(10, 20), 2., "model_score"))
    with pytest.raises(ValueError, match="mixed score"):
        measure_unit_organisation(UnitStream(occurrences=events, **metadata(["a"])), gap_seconds=0, overlap_policy="score_ranked")


@pytest.mark.parametrize("level", ["primitive", "episode", "assembly", "context"])
def test_same_measurement_consumer_is_independent_of_unit_level(level):
    stream = UnitStream(occurrences=(UnitOccurrence("one", "a", FrameSpan(0, 10)),
                                     UnitOccurrence("two", "a", FrameSpan(10, 20))), **metadata(["a"], level=level))
    counts = measure_unit_organisation(stream, gap_seconds=0)["counts"]
    assert counts["self_pairs"] == 1 and counts["calls"] == 2
    assert metric_values(counts)["mean_run_calls"] == 2


def test_valid_upstream_large_coordinates_have_honest_consumer_limit():
    stream = UnitStream(occurrences=(), **metadata([], spans=(FrameSpan(0, 2 ** 65),)))
    with pytest.raises(ValueError, match="int64"):
        measure_unit_organisation(stream, gap_seconds=0)


def test_staged_grammar_assembly_streams_enter_common_consumers():
    from fpcode.behavior.analysis.execution import measure_occurrence_execution
    from fpcode.behavior.episodes.grammar import GrammarEpisodeDetector
    from fpcode.behavior.episodes.staged import StagedEpisodeModel
    from fpcode.contracts.composition import GrammarRule, GrammarSpecification, SequenceRule, TerminalMatch

    cat = (UnitType('0'), UnitType('1'), UnitType('X'))
    prim_calls = (
        UnitOccurrence('p0', '0', FrameSpan(0, 5)),
        UnitOccurrence('p1', '1', FrameSpan(5, 10)),
        UnitOccurrence('p2', 'X', FrameSpan(10, 15)),
    )
    primitives = UnitStream(
        session=SESSION,
        level='primitive',
        model_id='prim_det',
        clock='video',
        fps=10.0,
        valid_intervals=(FrameSpan(0, 20),),
        catalog=cat,
        occurrences=prim_calls,
    )

    staged_model = StagedEpisodeModel.create_linear_routine(
        model_id='staged_model',
        routine_name='routine_01',
        phase_names=('p0', 'p1'),
        primitive_catalog=cat,
        phase_emissions={'p0': {'0': 1.0}, 'p1': {'1': 1.0}},
        background_emissions={'X': 1.0},
        allow_immediate_reentry=True,
    )
    staged_res = staged_model.decode_stream(primitives)

    ep_rule = GrammarRule('gram_ep', SequenceRule((TerminalMatch('0'), TerminalMatch('1'))))
    grammar_detector = GrammarEpisodeDetector(GrammarSpecification('ep_spec', (ep_rule,)))
    grammar_res = grammar_detector.detect(primitives)

    for res in (staged_res, grammar_res):
        org = measure_unit_organisation(res.stream, gap_seconds=0)
        assert org['counts']['calls'] == 1
        exec_df = measure_occurrence_execution(res)
        assert len(exec_df) == 1
        assert exec_df.loc[0, 'completion_status'] == 'censored'
        assert bool(exec_df.loc[0, 'left_censored']) is True
