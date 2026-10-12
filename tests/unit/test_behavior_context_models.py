"""Independent numerical tests for EventContextHMM."""

import numpy as np

from fpcode.behavior.context.models import EventContextHMM
from fpcode.contracts import (
    BehaviorResult,
    FrameSpan,
    SessionKey,
    UnitOccurrence,
    UnitStream,
    UnitType,
)

SESSION = SessionKey("rec_ctx", "mouse_ctx")


def test_event_context_hmm_infers_posteriors_and_preserves_lower_units():
    # 2 context states: "Exploration" (prefers P0) and "Grooming" (prefers P1)
    catalog = (UnitType("P0"), UnitType("P1"))
    log_init = np.log([0.5, 0.5])
    # Sticky transitions: 0.9 stay, 0.1 switch
    log_trans = np.log([[0.9, 0.1], [0.1, 0.9]])
    # Emissions: C0 -> 0.9 P0, 0.1 P1; C1 -> 0.1 P0, 0.9 P1
    log_emiss = np.log([[0.9, 0.1], [0.1, 0.9]])

    model = EventContextHMM(
        model_id="test_ctx_hmm",
        n_contexts=2,
        context_names=("Exploration", "Grooming"),
        unit_catalog=catalog,
        log_initial=log_init,
        log_transition=log_trans,
        log_emissions=log_emiss,
        unmatched_time_policy="break",
    )

    # 4 primitive events: P0, P0, P1, P1
    calls = (
        UnitOccurrence("c0", "P0", FrameSpan(0, 10)),
        UnitOccurrence("c1", "P0", FrameSpan(10, 20)),
        UnitOccurrence("c2", "P1", FrameSpan(20, 30)),
        UnitOccurrence("c3", "P1", FrameSpan(30, 40)),
    )
    stream = UnitStream(
        session=SESSION,
        level="primitive",
        model_id="prim_det",
        clock="video",
        fps=10.0,
        valid_intervals=(FrameSpan(0, 50),),
        catalog=catalog,
        occurrences=calls,
    )

    result = model.infer(stream)
    assert isinstance(result, BehaviorResult)
    assert result.level == "context"
    assert len(result.occurrences) == 4

    # First two should be Exploration, next two Grooming
    assert result.occurrences[0].type_id == "Exploration"
    assert result.occurrences[1].type_id == "Exploration"
    assert result.occurrences[2].type_id == "Grooming"
    assert result.occurrences[3].type_id == "Grooming"

    # Intervals match underlying events exactly (does not fill gaps!)
    assert result.occurrences[0].interval == FrameSpan(0, 10)
    assert result.occurrences[3].interval == FrameSpan(30, 40)

    # Posteriors sidecar
    post = result.posteriors
    assert post is not None
    assert post.observation_coordinate == "event"
    assert post.probabilities.shape == (4, 2)
    assert np.allclose(post.probabilities.sum(axis=1), 1.0)


def test_context_unmatched_time_policy():
    # Gap between c0 [0, 5) and c1 [15, 20)
    catalog = (UnitType("P0"), UnitType("P1"))
    log_init = np.log([0.5, 0.5])
    log_trans = np.log([[0.9, 0.1], [0.1, 0.9]])
    log_emiss = np.log([[0.9, 0.1], [0.1, 0.9]])

    model_break = EventContextHMM(
        model_id="ctx_break",
        n_contexts=2,
        context_names=("C0", "C1"),
        unit_catalog=catalog,
        log_initial=log_init,
        log_transition=log_trans,
        log_emissions=log_emiss,
        unmatched_time_policy="break",
    )

    calls = (
        UnitOccurrence("c0", "P0", FrameSpan(0, 5)),
        UnitOccurrence("c1", "P0", FrameSpan(15, 20)),
    )
    stream = UnitStream(
        session=SESSION,
        level="primitive",
        model_id="prim_det",
        clock="video",
        fps=10.0,
        valid_intervals=(FrameSpan(0, 30),),
        catalog=catalog,
        occurrences=calls,
    )

    res_break = model_break.infer(stream)
    # Under break, c1 is evaluated as start of a new independent block
    assert len(res_break.occurrences) == 2
    # Physical gap [5, 15) is NOT filled into a context occurrence!
    assert res_break.occurrences[0].interval == FrameSpan(0, 5)
    assert res_break.occurrences[1].interval == FrameSpan(15, 20)


def test_context_blocks_vocabulary_unknowns_and_frozen_inputs():
    from dataclasses import replace

    import pytest
    catalog = (UnitType('A'), UnitType('B'))
    model = EventContextHMM('context', 2, ('C0','C1'), catalog, np.log([.6,.4]),
                            np.log([[.9,.1],[.2,.8]]), np.log([[.8,.2],[.1,.9]]))
    calls = (UnitOccurrence('a','A',FrameSpan(0,2)), UnitOccurrence('b','B',FrameSpan(4,6)))
    stream = UnitStream(SESSION,'episode','lower','video',10.,(FrameSpan(0,6),),catalog,calls)
    before = stream.occurrences, stream.catalog, stream.valid_intervals
    separate = math_log(.6*.8+.4*.1) + math_log(.6*.2+.4*.9)
    assert np.isclose(model.score(stream), separate)
    joined = replace(model, unmatched_time_policy='event_steps')
    expected = sum(np.exp(joined.log_initial[i]+joined.log_emissions[i,0]
                          +joined.log_transition[i,j]+joined.log_emissions[j,1]) for i in range(2) for j in range(2))
    assert np.isclose(joined.score(stream), math_log(expected))
    result = joined.infer(stream)
    assert result.diagnostics['log_likelihood'] == joined.score(stream)
    assert [o.interval for o in result.occurrences] == [c.interval for c in calls]
    assert (stream.occurrences, stream.catalog, stream.valid_intervals) == before
    permutation = replace(joined, unit_catalog=catalog[::-1], log_emissions=joined.log_emissions[:,::-1])
    assert np.isclose(permutation.score(stream), joined.score(stream))
    np.testing.assert_allclose(permutation.infer(stream).posteriors.probabilities, result.posteriors.probabilities)
    divided = replace(stream, valid_intervals=(FrameSpan(0,2),FrameSpan(4,6)))
    assert np.isclose(joined.score(divided), separate)
    other = replace(divided, session=SessionKey('second_recording','second_mouse'))
    assert joined.score(other) == joined.score(divided)
    empty = replace(stream, occurrences=())
    assert joined.score(empty) == 0 and joined.infer(empty).posteriors.probabilities.shape == (0,2)
    unknown = replace(stream, catalog=catalog+(UnitType('U'),), occurrences=(UnitOccurrence('u','U',FrameSpan(0,2)),))
    with pytest.raises(ValueError, match='unknown held-out'):
        joined.infer(unknown)
    assert replace(joined, unknown_type_policy='uninformative').score(unknown) == pytest.approx(0.)
    impossible = replace(joined, log_initial=np.array([0.,-np.inf]), log_emissions=np.array([[0.,-np.inf],[-np.inf,0.]]))
    bad = replace(stream, occurrences=(UnitOccurrence('b','B',FrameSpan(0,2)),))
    assert impossible.score(bad) == -np.inf
    with pytest.raises(ValueError, match='impossible'):
        impossible.infer(bad)


def math_log(value):
    import math
    return math.log(value)
