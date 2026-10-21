"""Unit tests for BehaviorResult and composition contracts."""

import numpy as np
import pytest

from fpcode.behavior.compatibility import (
    StreamRequirements,
    inspect_compatibility,
    require_compatible,
)
from fpcode.contracts import (
    BehaviorResult,
    CompletionEvidence,
    FrameSpan,
    GrammarRule,
    GrammarSpecification,
    PhaseRecord,
    SequenceRule,
    SessionKey,
    StatePosteriorRecord,
    TerminalMatch,
    UnitOccurrence,
    UnitStream,
    UnitType,
)

SESSION = SessionKey("rec1", "mouse1")


def _stream():
    return UnitStream(
        session=SESSION,
        level="episode",
        model_id="test_model",
        clock="video",
        fps=30.0,
        valid_intervals=(FrameSpan(0, 100),),
        catalog=(UnitType("A"), UnitType("B")),
        occurrences=(
            UnitOccurrence("occ_1", "A", FrameSpan(10, 20), score=0.8, score_kind="model_score"),
            UnitOccurrence("occ_2", "B", FrameSpan(30, 50), score=0.9, score_kind="model_score"),
        ),
    )


def test_plain_unit_stream_rejects_completion_requirement():
    stream = _stream()
    req = StreamRequirements(required_information=("completion",))
    report = inspect_compatibility(stream, req)
    assert not report.compatible
    assert any("required information is unavailable in UnitStream: completion" in r for r in report.reasons)
    with pytest.raises(ValueError, match="required information is unavailable"):
        require_compatible(stream, req)


def test_behavior_result_with_completion_satisfies_requirement():
    stream = _stream()
    completion = (
        CompletionEvidence("occ_1", observed_entry=True, observed_exit=True, final_phase_reached=True, status="completed"),
        CompletionEvidence("occ_2", observed_entry=True, observed_exit=False, final_phase_reached=False, status="censored", right_censored=True),
    )
    result = BehaviorResult(stream=stream, completion=completion)
    assert "completion" in result.available_capabilities()
    req = StreamRequirements(required_information=("completion",))
    report = inspect_compatibility(result, req)
    assert report.compatible


def test_behavior_result_validates_foreign_occurrence_references():
    stream = _stream()
    bad_completion = (CompletionEvidence("foreign_occ", status="completed"),)
    with pytest.raises(ValueError, match="not in stream occurrences"):
        BehaviorResult(stream=stream, completion=bad_completion)

    bad_phase = (PhaseRecord("foreign_occ", "p1", 0, FrameSpan(10, 15)),)
    with pytest.raises(ValueError, match="not in stream occurrences"):
        BehaviorResult(stream=stream, phases=bad_phase)


def test_state_posterior_record_validates_probabilities_and_row_sums():
    catalog = (UnitType("A"), UnitType("B"))
    probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    valid = np.array([True, True])
    intervals = (FrameSpan(10, 20), FrameSpan(30, 50))
    rec = StatePosteriorRecord(
        observation_coordinate="event",
        state_catalog=catalog,
        probabilities=probs,
        valid_mask=valid,
        sample_intervals=intervals,
        session=SESSION,
        model_id="test_model",
    )
    assert rec.probabilities.shape == (2, 2)

    # Bad sum
    bad_probs = np.array([[0.7, 0.4], [0.2, 0.8]])
    with pytest.raises(ValueError, match="sum to 1"):
        StatePosteriorRecord(
            observation_coordinate="event",
            state_catalog=catalog,
            probabilities=bad_probs,
            valid_mask=valid,
            sample_intervals=intervals,
            session=SESSION,
            model_id="test_model",
        )


def test_grammar_specification_rejects_empty_and_duplicate_rules():
    t_a = TerminalMatch("A")
    t_b = TerminalMatch("B")
    seq = SequenceRule((t_a, t_b))
    rule1 = GrammarRule("rule_1", seq)
    spec = GrammarSpecification("grammar_1", (rule1,))
    assert len(spec.rules) == 1

    with pytest.raises(ValueError, match="requires at least one rule"):
        GrammarSpecification("empty", ())

    with pytest.raises(ValueError, match="identities must be unique"):
        GrammarSpecification("dupe", (rule1, GrammarRule("rule_1", t_a)))


def test_rich_references_parents_posteriors_and_owned_metadata():
    from dataclasses import replace

    from fpcode.contracts.behavior_results import CompositionDerivation, ScopedOccurrenceReference
    target = _stream()
    child = UnitOccurrence('child','A',FrameSpan(10,15))
    source = replace(target, level='primitive', model_id='raw', occurrences=(child,))
    ref = ScopedOccurrenceReference(SESSION,'raw','child')
    phase = PhaseRecord('occ_1','p',0,FrameSpan(10,15),ref)
    result = BehaviorResult(target, phases=(phase,), source_streams=(source,))
    assert 'phases' in result.available_capabilities()
    with pytest.raises(ValueError, match='resolve'):
        BehaviorResult(target, phases=(phase,))
    with pytest.raises(ValueError, match='phase orders'):
        BehaviorResult(target, phases=(phase,phase), source_streams=(source,))
    derivation = CompositionDerivation('rule','deriv',(ref,),-.5, interval=FrameSpan(10,20), parent_occurrence_id='occ_1')
    BehaviorResult(target, derivations=(derivation,), source_streams=(source,))
    with pytest.raises(ValueError, match='identities must be unique'):
        BehaviorResult(target, derivations=(derivation,derivation), source_streams=(source,))
    with pytest.raises(ValueError, match='parent not'):
        BehaviorResult(target, derivations=(replace(derivation,parent_occurrence_id='missing'),), source_streams=(source,))
    posterior = StatePosteriorRecord('event',target.catalog,np.array([[.8,.2]]),np.array([True]),
                                     (FrameSpan(10,20),),SESSION,target.model_id,clock='video')
    BehaviorResult(target,posteriors=posterior)
    for invalid in (replace(posterior,clock='other'), replace(posterior,model_id='other'),
                    replace(posterior,sample_intervals=(FrameSpan(100,110),))):
        with pytest.raises(ValueError, match='model/clock|outside'):
            BehaviorResult(target,posteriors=invalid)
    data = {'nested': {'values': [1,2]}, 'array': np.array([3.,4.])}
    owned = BehaviorResult(target,diagnostics=data)
    data['nested']['values'][0] = 99
    data['array'][0] = 99
    assert owned.diagnostics['nested']['values'] == (1,2)
    assert owned.diagnostics['array'][0] == 3.
    with pytest.raises(TypeError):
        owned.diagnostics['nested']['changed'] = True
    with pytest.raises(ValueError, match='fitted objects'):
        BehaviorResult(target,diagnostics={'fitter': object()})


def test_feature_association_matches_actual_coordinates_and_unsigned_order():
    from fpcode.contracts import SignalMetadata, SignalSeries, TimeAxis
    from fpcode.contracts.behavior_results import FeatureAssociation
    series = SignalSeries(TimeAxis(np.array([10.,10.1]),'s','video'), np.ones((2,1)), (SignalMetadata('speed'),))
    normalized = SignalSeries(TimeAxis(np.array([20.,20.1]),'s','video'), np.zeros((2,1)),series.signals)
    with pytest.raises(ValueError, match='preserve raw coordinates'):
        FeatureAssociation(series,SESSION,'arena',normalized_series=normalized)
    with pytest.raises(ValueError, match='increasing'):
        FeatureAssociation(series,SESSION,'arena',native_frames=np.array([2,1],dtype=np.uint64))
