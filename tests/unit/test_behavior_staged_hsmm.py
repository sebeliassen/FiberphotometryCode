"""Independent numerical tests for sequence inference and staged HSMM episodes."""

import math

import numpy as np

from fpcode.behavior.episodes.staged import StagedEpisodeModel
from fpcode.contracts import (
    BehaviorResult,
    FrameSpan,
    SessionKey,
    UnitOccurrence,
    UnitStream,
    UnitType,
)
from fpcode.statistics.sequence_inference import (
    compute_survival_log_pmfs,
    forward_backward_hmm,
    viterbi_hmm,
    viterbi_hsmm,
)


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), float("-inf"))

SESSION = SessionKey("rec_hsmm", "mouse_hsmm")


def test_hmm_enumerated_likelihood_marginals_and_map():
    # 2 states (0, 1), 3 observations
    # Initial: [0.6, 0.4]
    # Transition: [[0.7, 0.3], [0.2, 0.8]]
    # Emission for obs (t=0,1,2):
    # obs 0: [0.9, 0.1]
    # obs 1: [0.2, 0.8]
    # obs 2: [0.1, 0.9]
    init = np.array([0.6, 0.4])
    trans = np.array([[0.7, 0.3], [0.2, 0.8]])
    emiss = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]])

    # Enumerate all 2^3 = 8 paths by hand:
    # path = (s0, s1, s2), prob = init[s0]*emiss[0,s0] * trans[s0,s1]*emiss[1,s1] * trans[s1,s2]*emiss[2,s2]
    exact_probs = {}
    for s0 in (0, 1):
        for s1 in (0, 1):
            for s2 in (0, 1):
                p = (init[s0] * emiss[0, s0]) * (trans[s0, s1] * emiss[1, s1]) * (trans[s1, s2] * emiss[2, s2])
                exact_probs[(s0, s1, s2)] = p

    total_lik = sum(exact_probs.values())
    best_path = max(exact_probs, key=exact_probs.get)
    best_score = exact_probs[best_path]

    # Hand-computed marginals for t=1, state=1
    p_t1_s1 = sum(p for (s0, s1, s2), p in exact_probs.items() if s1 == 1) / total_lik

    # Now run forward_backward_hmm and viterbi_hmm
    fb = forward_backward_hmm(np.log(init), np.log(trans), np.log(emiss))
    assert math.isclose(np.exp(fb.log_likelihood), total_lik, rel_tol=1e-6)
    assert math.isclose(fb.posteriors[1, 1], p_t1_s1, rel_tol=1e-6)

    map_path, map_score = viterbi_hmm(np.log(init), np.log(trans), np.log(emiss))
    assert tuple(map_path) == best_path
    assert math.isclose(np.exp(map_score), best_score, rel_tol=1e-6)


def test_structural_zero_forbidden_transition_has_zero_mass():
    # State 0 cannot transition to State 1: trans[0, 1] = 0 (log = -inf)
    init = np.array([1.0, 0.0])
    trans = np.array([[1.0, 0.0], [0.5, 0.5]])  # trans[0, 1] is exactly 0
    emiss = np.array([[0.5, 0.5], [0.1, 0.9]])   # obs 1 strongly suggests state 1, but path is forbidden!

    def _safe_log(x):
        return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), float("-inf"))
    log_trans = _safe_log(trans)
    fb = forward_backward_hmm(_safe_log(init), log_trans, _safe_log(emiss))
    # At t=1, posterior for state 1 must be exactly 0
    assert fb.posteriors[1, 1] == 0.0
    assert not math.isnan(fb.posteriors[1, 1])


def test_survival_pmf_fresh_start_boundary():
    # PMF {0.4, 0.6} for D in {1, 2}
    # Survival P(D >= 1) = 1.0 (log = 0.0)
    # Survival P(D >= 2) = 0.6 (log = ln(0.6))
    pmf = np.array([[0.4, 0.6]])
    log_pmf = np.log(pmf)
    surv = compute_survival_log_pmfs(log_pmf)

    assert math.isclose(np.exp(surv[0, 0]), 1.0, rel_tol=1e-6)
    assert math.isclose(np.exp(surv[0, 1]), 0.6, rel_tol=1e-6)


def test_hsmm_enumerated_segmentations():
    # 2 states: 0 and 1. 3 observations. Duration support {1, 2}.
    # Initial: [1.0, 0.0] (must start in state 0).
    # Transition: trans[0, 1] = 1.0, trans[1, 0] = 1.0, self-transitions = 0.
    # State 0 duration PMF: [0.5, 0.5].
    # State 1 duration PMF: [0.5, 0.5].
    init = np.array([1.0, 0.0])
    trans = np.array([[0.0, 1.0], [1.0, 0.0]])
    dur_pmf = np.array([[0.5, 0.5], [0.5, 0.5]])
    emiss = np.ones((3, 2)) * 0.5  # uniform emissions

    log_init = _safe_log(init)
    log_trans = _safe_log(trans)
    log_dur = _safe_log(dur_pmf)
    log_emiss = _safe_log(emiss)

    # Possible segmentations starting in state 0 for T=3:
    # 1. State 0 dur 1 (t=0..1), then State 1 dur 2 (t=1..3).
    #    Score (uncensored): init[0]*dur[0,1]*emiss * trans[0,1]*dur[1,2]*emiss^2
    # 2. State 0 dur 2 (t=0..2), then State 1 dur 1 (t=2..3).
    # Both are legal and have equal probability!
    res = viterbi_hsmm(log_init, log_trans, log_emiss, log_dur, right_censored=False)
    assert len(res.segments) == 2
    assert res.segments[0].state == 0
    assert res.segments[1].state == 1
    assert res.segments[0].duration + res.segments[1].duration == 3


def test_staged_episode_immediate_reentry_creates_two_invocations():
    # Linear routine with 2 phases: "entry" and "exit"
    # Primitive tokens: "A" (entry), "B" (exit)
    # Sequence of primitives: [A, B, A, B]
    # Under immediate reentry, this MUST decode into TWO distinct episode occurrences (ep_0 and ep_1), NOT one long episode!
    cat = (UnitType("A"), UnitType("B"), UnitType("X"))
    model = StagedEpisodeModel.create_linear_routine(
        model_id="test_staged",
        routine_name="routine_AB",
        phase_names=("entry", "exit"),
        primitive_catalog=cat,
        phase_emissions={
            "entry": {"A": 1.0},
            "exit": {"B": 1.0},
        },
        background_emissions={"X": 1.0},
        allow_immediate_reentry=True,
        mode="explicit_duration_hsmm",
    )

    # Input primitive stream with calls: A at [0, 5), B at [5, 10), A at [10, 15), B at [15, 20)
    calls = (
        UnitOccurrence("c0", "A", FrameSpan(0, 5)),
        UnitOccurrence("c1", "B", FrameSpan(5, 10)),
        UnitOccurrence("c2", "A", FrameSpan(10, 15)),
        UnitOccurrence("c3", "B", FrameSpan(15, 20)),
        UnitOccurrence("c4", "X", FrameSpan(20, 25)),
    )
    stream = UnitStream(
        session=SESSION,
        level="primitive",
        model_id="test_det",
        clock="video",
        fps=10.0,
        valid_intervals=(FrameSpan(0, 30),),
        catalog=cat,
        occurrences=calls,
    )

    result = model.decode_stream(stream)
    assert isinstance(result, BehaviorResult)
    # Must have 2 occurrences
    assert len(result.stream.occurrences) == 2
    assert result.stream.occurrences[0].occurrence_id == "ep_0"
    assert result.stream.occurrences[1].occurrence_id == "ep_1"
    assert (result.stream.occurrences[0].interval.start, result.stream.occurrences[0].interval.stop) == (0, 10)
    assert (result.stream.occurrences[1].interval.start, result.stream.occurrences[1].interval.stop) == (10, 20)

    # The first entry is at a recording boundary; the second is observed reentry.
    assert result.completion[0].status == "censored"
    assert result.completion[0].left_censored
    assert result.completion[1].status == "completed"
    assert result.completion[0].final_phase_reached
    assert result.completion[1].final_phase_reached
    assert all(c.score is None for c in result.completion)
    assert result.diagnostics["log_likelihood"] == model.score(stream)


def test_staged_episode_boundary_censoring():
    # Sequence stops after first phase: [A]
    # Does not reach exit phase before recording end -> censored!
    cat = (UnitType("A"), UnitType("B"), UnitType("X"))
    model = StagedEpisodeModel.create_linear_routine(
        model_id="test_staged",
        routine_name="routine_AB",
        phase_names=("entry", "exit"),
        primitive_catalog=cat,
        phase_emissions={"entry": {"A": 1.0}, "exit": {"B": 1.0}},
        background_emissions={"X": 1.0},
        allow_immediate_reentry=True,
    )
    calls = (UnitOccurrence("c0", "A", FrameSpan(0, 5)),)
    stream = UnitStream(
        session=SESSION,
        level="primitive",
        model_id="test_det",
        clock="video",
        fps=10.0,
        valid_intervals=(FrameSpan(0, 10),),
        catalog=cat,
        occurrences=calls,
    )
    result = model.decode_stream(stream)
    assert len(result.stream.occurrences) == 1
    comp = result.completion[0]
    assert comp.status == "censored"
    assert comp.right_censored
    assert not comp.final_phase_reached


def test_hsmm_matches_independent_enumeration_with_and_without_censoring():
    from fpcode.statistics.sequence_inference import forward_hsmm
    initial = np.array([.6, .4])
    transitions = np.array([[0., 1.], [1., 0.]])
    duration = np.array([[.3, .7], [.8, .2]])
    emissions = np.array([[.8, .2], [.2, .8], [0., 1.], [.7, .3], [.4, .6]])
    for length in range(1, 6):
        for censored in (False, True):
            candidates = []

            def enumerate_segments(start, previous, mass, segments):
                if start == length:
                    candidates.append((mass, segments))
                    return
                for state in range(2):
                    entry = initial[state] if previous is None else transitions[previous, state]
                    for observed_duration in (1, 2):
                        stop = start + observed_duration
                        if stop > length:
                            continue
                        dwell = (duration[state, observed_duration-1:].sum()
                                 if censored and stop == length else duration[state, observed_duration-1])
                        weight = entry*dwell*np.prod(emissions[start:stop, state])
                        if weight:
                            enumerate_segments(stop, state, mass*weight,
                                               segments+((state, start, stop),))

            enumerate_segments(0, None, 1., ())
            args = (_safe_log(initial), _safe_log(transitions), _safe_log(emissions[:length]), _safe_log(duration))
            actual = viterbi_hsmm(*args, right_censored=censored)
            forward, _ = forward_hsmm(*args, right_censored=censored)
            total = sum(m for m, _ in candidates)
            maximum = max(m for m, _ in candidates)
            assert math.isclose(math.exp(actual.log_likelihood), total, rel_tol=1e-12)
            assert math.isclose(forward, actual.log_likelihood, abs_tol=1e-12)
            assert math.isclose(math.exp(actual.best_score), maximum, rel_tol=1e-12)
            selected = tuple((s.state, s.start, s.stop) for s in actual.segments)
            assert any(selected == path and math.isclose(mass, maximum, rel_tol=1e-12) for mass, path in candidates)
            assert np.all(emissions[np.arange(length), actual.map_states] > 0)


def test_inference_rejects_invalid_probability_tables_and_resource_caps():
    import pytest

    from fpcode.statistics.sequence_inference import forward_hsmm
    valid = (np.log([.5, .5]), np.array([[-np.inf, 0.], [0., -np.inf]]), np.zeros((2, 2)), np.zeros((2, 1)))
    with pytest.raises(ValueError, match='normalized'):
        forward_hsmm(np.log([.3, .3]), *valid[1:])
    with pytest.raises(ValueError, match='duration PMFs'):
        forward_hsmm(*valid[:3], np.log([[.2], [1.]]))
    with pytest.raises(ValueError, match='memory budget'):
        forward_hsmm(*valid, memory_budget_bytes=1)
    with pytest.raises(ValueError, match='work budget'):
        viterbi_hsmm(*valid, max_work=1)
