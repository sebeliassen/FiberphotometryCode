"""Independent numerical tests for sequence inference and staged HSMM episodes."""

import math
from dataclasses import replace

import numpy as np
import pytest

from fpcode.behavior.episodes.staged import StagedEpisodeModel, fit_staged_model, gaussian_log_emissions
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


def test_gaussian_feature_log_density_against_independent_formula():
    values, means, variances = np.array([[1., -2.]]), np.array([[0., 0.], [1., -1.]]), np.array([[4., 1.], [1., 9.]])
    expected = [-.5*sum(math.log(2*math.pi*v)+(x-m)**2/v for x, m, v in zip(values[0], mu, var))
                for mu, var in zip(means, variances)]
    np.testing.assert_allclose(gaussian_log_emissions(values, means, variances)[0], expected, atol=1e-12)


def test_gaussian_feature_density_preserves_small_residual_at_large_offset():
    result = gaussian_log_emissions([[1e9+1.]], [[1e9]], [[1.]])
    np.testing.assert_allclose(result, [[-.5*(math.log(2*math.pi)+1.)]], atol=1e-12)
    with pytest.raises(ValueError, match='float64 range'):
        gaussian_log_emissions([[1e200]], [[0.]], [[1.]])


@pytest.mark.parametrize('offset', [0., 1e9])
def test_gaussian_fit_parameters_match_assigned_observation_mle(offset):
    catalog = (UnitType('X'), UnitType('A'))
    model = StagedEpisodeModel.create_linear_routine('configured', 'A', ('a',), catalog,
        {'a': {'A': 1.}}, {'X': 1.}, allow_immediate_reentry=False,
        phase_durations={'a': np.ones(3)/3}, background_duration=np.ones(3)/3)
    model = replace(model, feature_names=('x',), feature_means=np.array([[0.], [5.]])+offset,
                    feature_variances=np.ones((2, 1)))
    calls = tuple(UnitOccurrence(str(i), t, FrameSpan(i, i+1)) for i, t in enumerate(('X', 'X', 'A', 'A', 'X', 'X')))
    stream = UnitStream(SessionKey('gaussian', 'm'), 'primitive', 'source', 'native', 10.,
                        (FrameSpan(0, 6),), catalog, calls)
    fit = fit_staged_model(model, (stream,), model_id='learned', source_model_id='source',
            max_iterations=1, training_features=(np.array([[0.], [2.], [4.], [8.], [0.], [2.]])+offset,), variance_floor=.05)
    np.testing.assert_allclose(fit.model.feature_means[:, 0], np.array([1., 6.])+offset, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fit.model.feature_variances[:, 0], [1., 4.])
    assert 'Gaussian_means' in fit.diagnostics['learned_parameters']
    assert fit.diagnostics['estimated_peak_array_bytes'] > 6*8


@pytest.mark.parametrize('weight', [True, -1., float('nan'), [1.], np.array([1.])])
def test_feature_weights_require_finite_nonnegative_real_scalar(weight):
    model = StagedEpisodeModel.create_linear_routine('configured', 'A', ('a',), (UnitType('A'),),
        {'a': {'A': 1.}}, {'A': 1.}, allow_immediate_reentry=False)
    with pytest.raises(ValueError, match='feature weight'):
        replace(model, feature_weight=weight)


def test_feature_input_changes_hsmm_detection_and_background_can_persist():
    catalog = (UnitType("X"),)
    initial = StagedEpisodeModel.create_linear_routine("initial", "routine", ("a", "b"), catalog,
              {"a": {"X": 1.}, "b": {"X": 1.}}, {"X": 1.}, extra_background=True,
              entry_probability=.01, phase_durations={"a": np.array([1.]), "b": np.array([1.])})
    calls = tuple(UnitOccurrence(str(i), "X", FrameSpan(i*5, (i+1)*5)) for i in range(5))
    stream = UnitStream(SessionKey("feature", "m"), "primitive", "source", "native", 10.,
                        (FrameSpan(0, 25),), catalog, calls)
    assert not initial.decode_stream(stream).occurrences
    mixed = replace(initial, feature_names=("speed",), feature_means=np.array([[0.], [5.], [5.], [0.]]),
                    feature_variances=np.full((4, 1), .05))
    result = mixed.decode_stream(stream, features=np.array([[0.], [0.], [5.], [5.], [0.]]))
    assert [o.interval for o in result.occurrences] == [FrameSpan(10, 20)]
    with pytest.raises(ValueError, match="aligned"):
        mixed.decode_stream(stream, features=np.zeros((4, 1)))


def test_duration_learning_counts_only_observed_interior_segments():
    catalog = (UnitType("X"), UnitType("A"))
    model = StagedEpisodeModel.create_linear_routine("configured", "A", ("a",), catalog,
              {"a": {"A": 1.}}, {"X": 1.}, allow_immediate_reentry=False,
              phase_durations={"a": np.ones(3)/3}, background_duration=np.ones(3)/3)
    calls = tuple(UnitOccurrence(str(i), t, FrameSpan(i*2, (i+1)*2))
                  for i, t in enumerate(("X", "X", "A", "A", "X", "X", "A", "A", "X", "X")))
    stream = UnitStream(SessionKey("duration", "m"), "primitive", "source", "native", 10.,
                        (FrameSpan(0, 20),), catalog, calls)
    fit = fit_staged_model(model, (stream,), model_id="learned", source_model_id="source",
                          max_iterations=1, learn_durations=True, pseudocount=.5)
    np.testing.assert_allclose(np.exp(fit.model.log_duration_pmfs[0]), [1/9, 7/9, 1/9])
    np.testing.assert_allclose(np.exp(fit.model.log_duration_pmfs[1]), [1/15, 13/15, 1/15])
    assert fit.diagnostics["duration_update_count"] == 1
    np.testing.assert_array_equal(fit.diagnostics["interior_duration_counts"], [[0, 1, 0], [0, 2, 0]])

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


def _fit_fixture(*, durations=None):
    catalog = (UnitType("A"), UnitType("B"), UnitType("X"))
    model = StagedEpisodeModel.create_linear_routine(
        "configured_AB", "AB", ("entry", "exit"), catalog,
        {"entry": {"A": .8, "B": .2}, "exit": {"A": .2, "B": .8}}, {"X": 1.},
        phase_durations=durations,
    )
    return model, catalog


def _training_stream(catalog, labels, *, recording="fit_recording", starts=None):
    starts = tuple(range(len(labels))) if starts is None else starts
    return UnitStream(
        SessionKey(recording, "fit_mouse"), "primitive", "frozen_primitives", "video", 10.,
        (FrameSpan(0, max(starts, default=0) + 1),), catalog,
        tuple(UnitOccurrence(f"b{i}", label, FrameSpan(start, start + 1))
              for i, (label, start) in enumerate(zip(labels, starts))),
    )


def test_hard_em_reestimation_matches_independent_categorical_and_transition_counts():
    model, catalog = _fit_fixture()
    stream = _training_stream(catalog, ("A", "B", "A", "B", "A", "B", "X"))
    original = tuple(array.copy() for array in (model.log_emissions, model.log_transition,
                                               model.log_initial, model.log_duration_pmfs))
    result = fit_staged_model(model, (stream,), model_id="learned_AB", source_model_id=stream.model_id,
                              max_iterations=1, pseudocount=.5)
    # Three entries all emit A, three exits all emit B; each row has prior mass .5.
    np.testing.assert_allclose(np.exp(result.model.log_emissions),
                               [[0., 0., 1.], [3.4 / 3.5, .1 / 3.5, 0.], [.1 / 3.5, 3.4 / 3.5, 0.]])
    # Final-phase transitions include two reentries and one observed exit.
    np.testing.assert_allclose(np.exp(result.model.log_transition[-1]), [1.25 / 3.5, 2.25 / 3.5, 0.])
    np.testing.assert_array_equal(result.model.log_initial, model.log_initial)
    np.testing.assert_array_equal(result.model.log_duration_pmfs, model.log_duration_pmfs)
    for array, before in zip((model.log_emissions, model.log_transition,
                              model.log_initial, model.log_duration_pmfs), original):
        np.testing.assert_array_equal(array, before)
    assert np.array_equal(np.isneginf(result.model.log_emissions), np.isneginf(model.log_emissions))
    assert np.array_equal(np.isneginf(result.model.log_transition), np.isneginf(model.log_transition))
    assert not result.model.log_emissions.flags.writeable
    with pytest.raises(TypeError):
        result.diagnostics["seed"] = 7
    diagnostics = result.diagnostics
    assert diagnostics["emission_observation_counts"] == ((0., 0., 1.), (3., 0., 0.), (0., 3., 0.))
    assert diagnostics["segment_transition_counts"] == ((0., 0., 0.), (0., 0., 3.), (1., 2., 0.))
    assert diagnostics["completed_iterations"] == 1
    assert diagnostics["updated_log_likelihood"] == result.model.score(stream)
    assert diagnostics["initial_log_likelihood"] == model.score(stream)
    assert diagnostics["updated_selected_path_score"] > diagnostics["initial_selected_path_score"]
    assert len(diagnostics["regularized_path_objective_history"]) == 2
    assert diagnostics["regularized_path_objective_history"][1] >= diagnostics["regularized_path_objective_history"][0]


def test_hard_em_preserves_gap_and_recording_barriers_in_transition_counts():
    model, catalog = _fit_fixture()
    # Every isolated A begins in entry. Concatenating would force exit/reentry,
    # but independent gaps/recordings provide no transition observations.
    first = _training_stream(catalog, ("A", "A"), starts=(0, 3))
    second = _training_stream(catalog, ("A",), recording="another_recording")
    result = fit_staged_model(model, (first, second), model_id="learned_gaps",
                              source_model_id=first.model_id, max_iterations=2)
    assert result.diagnostics["training_blocks"] == 3
    assert result.diagnostics["segment_transition_counts"] == ((0., 0., 0.),) * 3
    np.testing.assert_array_equal(result.model.log_transition, model.log_transition)
    # Unvisited exit/background emission rows retain the configured distributions.
    np.testing.assert_allclose(result.model.log_emissions[[0, 2]], model.log_emissions[[0, 2]])
    assert result.diagnostics["initial_boundary_segments"] == 3
    assert result.diagnostics["terminal_censored_segments"] == 3
    assert result.diagnostics["duration_update_count"] == 0


def test_hard_em_keeps_censored_boundary_duration_support_frozen():
    durations = {"entry": np.array([0., 0., 1.]), "exit": np.array([1.])}
    model, catalog = _fit_fixture(durations=durations)
    # The observed single A is a right-censored entry with D>=1; D=1 has zero PMF.
    stream = _training_stream(catalog, ("A",))
    result = fit_staged_model(model, (stream,), model_id="learned_censored",
                              source_model_id=stream.model_id, max_iterations=2)
    np.testing.assert_array_equal(result.model.log_duration_pmfs, model.log_duration_pmfs)
    assert np.isneginf(result.model.log_duration_pmfs[1, 0])
    assert result.diagnostics["initial_boundary_segments"] == 1
    assert result.diagnostics["terminal_censored_segments"] == 1
    with pytest.raises(ValueError, match="impossible training block"):
        fit_staged_model(model, (stream,), model_id="uncensored",
                         source_model_id=stream.model_id, right_censored=False)


def test_hard_em_retains_extremely_small_finite_categorical_support():
    model, catalog = _fit_fixture()
    log_emissions = model.log_emissions.copy()
    log_emissions[1, :2] = [0., -1000.]
    model = replace(model, log_emissions=log_emissions)
    stream = _training_stream(catalog, ("A",))
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        result = fit_staged_model(model, (stream,), model_id="small_supported_mass",
                                  source_model_id=stream.model_id, max_iterations=1)
    assert np.isfinite(result.model.log_emissions[1, 1])
    assert result.model.log_emissions[1, 1] == pytest.approx(-1000. - math.log(3.))
    assert np.isneginf(result.model.log_emissions[1, 2])


def test_hard_em_seeded_initialization_repeats_and_reports_actual_effort():
    model, catalog = _fit_fixture()
    stream = _training_stream(catalog, ("A", "B", "A", "B", "X"))
    kwargs = dict(model_id="seeded_fit", source_model_id=stream.model_id,
                  initialization="dirichlet", seed=17, max_iterations=10)
    first = fit_staged_model(model, (stream,), **kwargs)
    second = fit_staged_model(model, (stream,), **kwargs)
    np.testing.assert_array_equal(first.model.log_emissions, second.model.log_emissions)
    np.testing.assert_array_equal(first.model.log_transition, second.model.log_transition)
    assert first.diagnostics["log_likelihood_history"] == second.diagnostics["log_likelihood_history"]
    changed = fit_staged_model(model, (stream,), **{**kwargs, "seed": 18})
    assert first.diagnostics["initial_log_likelihood"] != changed.diagnostics["initial_log_likelihood"]
    assert first.diagnostics["initialization"] == "dirichlet"
    assert first.diagnostics["seed"] == 17
    assert first.diagnostics["converged"]
    assert first.diagnostics["completed_iterations"] < first.diagnostics["requested_iterations"]
    assert first.diagnostics["estimated_executed_work"] <= first.diagnostics["estimated_requested_work"]
    assert all(b >= a - 1e-12 for a, b in zip(first.diagnostics["regularized_path_objective_history"],
                                             first.diagnostics["regularized_path_objective_history"][1:]))


@pytest.mark.parametrize("field,value,match", [
    ("source_model_id", "wrong_primitives", "primitive model identity"),
    ("model_id", "configured_AB", "differ from initial"),
    ("max_iterations", 0, "positive integer"),
    ("max_observations", 1, "observation budget"),
    ("max_total_work", 1, "total work budget"),
    ("memory_budget_bytes", 1, "memory budget"),
    ("pseudocount", 0., "pseudocount"),
    ("pseudocount", np.nan, "pseudocount"),
    ("tolerance", -1., "tolerance"),
    ("seed", True, "seed"),
    ("initialization", "guessed", "initialization"),
])
def test_hard_em_rejects_invalid_scope_and_resource_settings(field, value, match):
    model, catalog = _fit_fixture()
    stream = _training_stream(catalog, ("A", "B"))
    kwargs = dict(model_id="fit_AB", source_model_id=stream.model_id)
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        fit_staged_model(model, (stream,), **kwargs)


def test_hard_em_rejects_ambiguous_catalog_clocks_duplicate_recordings_and_empty_inputs():
    model, catalog = _fit_fixture()
    stream = _training_stream(catalog, ("A",))
    kwargs = dict(model_id="fit_AB", source_model_id=stream.model_id)
    with pytest.raises(ValueError, match="duplicate training recording"):
        fit_staged_model(model, (stream, stream), **kwargs)
    other = _training_stream(catalog, ("A",), recording="different")
    with pytest.raises(ValueError, match="clock identity"):
        fit_staged_model(model, (stream, replace(other, clock="another_camera")), **kwargs)
    renamed = (UnitType("A", "different vocabulary description"), *catalog[1:])
    with pytest.raises(ValueError, match="catalogue differs"):
        fit_staged_model(model, (replace(stream, catalog=renamed),), **kwargs)
    with pytest.raises(ValueError, match="nonempty training"):
        fit_staged_model(model, (), **kwargs)
    with pytest.raises(ValueError, match="no primitive observations"):
        fit_staged_model(model, (replace(stream, occurrences=()),), **kwargs)
