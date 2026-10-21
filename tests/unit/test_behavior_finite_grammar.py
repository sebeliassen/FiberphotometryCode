"""Independent numerical tests for finite grammar composition."""

import math

import pytest

from fpcode.behavior.assemblies.grammar import GrammarAssemblyComposer
from fpcode.behavior.episodes.grammar import GrammarEpisodeDetector
from fpcode.contracts import (
    ChoiceRule,
    FrameSpan,
    GrammarRule,
    GrammarSpecification,
    RepeatRule,
    SequenceRule,
    SessionKey,
    TerminalMatch,
    UnitOccurrence,
    UnitStream,
    UnitType,
)
from fpcode.statistics.finite_composition import (
    match_rule_on_sequence,
)

SESSION = SessionKey("rec_gram", "mouse_gram")


def test_grammar_ambiguity_two_weighted_branches():
    # Identical terminal in choice branches weighted 0.25 and 0.75
    # Token "A": two derivations, summed mass 1.0, best derivation 0.75
    t1 = TerminalMatch("A")
    t2 = TerminalMatch("A")
    choice = ChoiceRule((t1, t2), weights=(0.25, 0.75))
    rule = GrammarRule("choice_rule", choice)

    matches = match_rule_on_sequence(rule, ("A",))
    assert len(matches) == 2
    scores = [m.score for m in matches]
    sorted_scores = sorted(scores)
    assert math.isclose(sorted_scores[0], 0.25, rel_tol=1e-9)
    assert math.isclose(sorted_scores[1], 0.75, rel_tol=1e-9)
    assert math.isclose(sum(scores), 1.0, rel_tol=1e-9)
    assert sum(scores) == pytest.approx(1.0)
    best_score = max(scores)
    assert best_score == pytest.approx(0.75)


def test_repeat_rule_min_max_boundaries():
    # Repeat "A" min 2, max 3 times
    rep = RepeatRule(TerminalMatch("A"), min_repeats=2, max_repeats=3)
    rule = GrammarRule("repeat_rule", rep)

    # 1 "A" -> 0 matches
    assert len(match_rule_on_sequence(rule, ("A",))) == 0

    # 2 "A"s -> matches length 2
    m2 = match_rule_on_sequence(rule, ("A", "A"))
    assert any(m.stop_pos - m.start_pos == 2 for m in m2)

    # 3 "A"s -> matches length 2 and length 3
    m3 = match_rule_on_sequence(rule, ("A", "A", "A"))
    lens = {m.stop_pos - m.start_pos for m in m3 if m.start_pos == 0}
    assert 2 in lens and 3 in lens
    # 4 is not matched (bounded at max 3)
    assert 4 not in lens


def test_shared_parser_at_primitive_and_episode_levels():
    # Sequence of two tokens: ("P1", "P2") -> episode
    # Then sequence of two episodes: ("E1", "E2") -> assembly
    ep_rule = GrammarRule("E_pair", SequenceRule((TerminalMatch("P1"), TerminalMatch("P2"))))
    ep_spec = GrammarSpecification("ep_grammar", (ep_rule,))
    ep_det = GrammarEpisodeDetector(ep_spec)

    prim_calls = (
        UnitOccurrence("p0", "P1", FrameSpan(0, 5)),
        UnitOccurrence("p1", "P2", FrameSpan(5, 10)),
    )
    prim_stream = UnitStream(
        session=SESSION,
        level="primitive",
        model_id="prim_det",
        clock="video",
        fps=10.0,
        valid_intervals=(FrameSpan(0, 20),),
        catalog=(UnitType("P1"), UnitType("P2")),
        occurrences=prim_calls,
    )
    ep_res = ep_det.detect(prim_stream)
    assert ep_res.level == "episode"
    assert len(ep_res.occurrences) == 1
    assert ep_res.occurrences[0].type_id == "E_pair"
    assert (ep_res.occurrences[0].interval.start, ep_res.occurrences[0].interval.stop) == (0, 10)

    # Scoped child references in derivation
    deriv = ep_res.derivations[0]
    assert len(deriv.child_references) == 2
    assert deriv.child_references[0].occurrence_id == "p0"
    assert deriv.child_references[1].occurrence_id == "p1"

    # Now assemble episodes into an assembly
    as_rule = GrammarRule("Assembly_double", SequenceRule((TerminalMatch("E_pair"), TerminalMatch("E_pair"))))
    as_spec = GrammarSpecification("as_grammar", (as_rule,))
    as_composer = GrammarAssemblyComposer(as_spec)

    ep_calls = (
        UnitOccurrence("ep0", "E_pair", FrameSpan(0, 10)),
        UnitOccurrence("ep1", "E_pair", FrameSpan(10, 20)),
    )
    ep_stream = UnitStream(
        session=SESSION,
        level="episode",
        model_id="ep_det",
        clock="video",
        fps=10.0,
        valid_intervals=(FrameSpan(0, 30),),
        catalog=(UnitType("E_pair"),),
        occurrences=ep_calls,
    )
    as_res = as_composer.compose(ep_stream)
    assert as_res.level == "assembly"
    assert len(as_res.occurrences) == 1
    assert as_res.occurrences[0].type_id == "Assembly_double"
    assert (as_res.occurrences[0].interval.start, as_res.occurrences[0].interval.stop) == (0, 20)
    as_deriv = as_res.derivations[0]
    assert len(as_deriv.child_references) == 2
    assert as_deriv.child_references[0].occurrence_id == "ep0"
    assert as_deriv.child_references[1].occurrence_id == "ep1"


def test_shared_children_dag_in_assemblies():
    # Two assembly definitions sharing the same child episode type "E_run"
    r1 = GrammarRule("A_run_then_turn", SequenceRule((TerminalMatch("E_run"), TerminalMatch("E_turn"))))
    r2 = GrammarRule("A_run_then_rest", SequenceRule((TerminalMatch("E_run"), TerminalMatch("E_rest"))))
    spec = GrammarSpecification("multi_as", (r1, r2))
    composer = GrammarAssemblyComposer(spec)

    ep_calls = (
        UnitOccurrence("e0", "E_run", FrameSpan(0, 10)),
        UnitOccurrence("e1", "E_turn", FrameSpan(10, 20)),
    )
    ep_stream = UnitStream(
        session=SESSION,
        level="episode",
        model_id="ep_det",
        clock="video",
        fps=10.0,
        valid_intervals=(FrameSpan(0, 30),),
        catalog=(UnitType("E_run"), UnitType("E_turn"), UnitType("E_rest")),
        occurrences=ep_calls,
    )
    res = composer.compose(ep_stream)
    # Only A_run_then_turn matches
    assert len(res.occurrences) == 1
    assert res.occurrences[0].type_id == "A_run_then_turn"


def test_pruning_retains_lengths_needed_by_a_sequence_and_marks_ambiguity():
    from fpcode.contracts.composition import OptionalRule
    # AB has two derivations: skip first A or skip second A, each with mass .25.
    rule = GrammarRule('optional_AB', SequenceRule((OptionalRule(TerminalMatch('A')),
                       OptionalRule(TerminalMatch('A')), TerminalMatch('B'))))
    matches = match_rule_on_sequence(rule, ('A', 'B'), max_derivations=1)
    full = [m for m in matches if m.start_pos == 0 and m.stop_pos == 2]
    assert len(full) == 1 and full[0].truncated
    assert full[0].score == pytest.approx(.25)
    assert full[0].trace
    repeat = GrammarRule('repeat', RepeatRule(TerminalMatch('A'), 2, 3))
    lengths = {m.stop_pos for m in match_rule_on_sequence(repeat, ('A',)*3, max_derivations=1) if m.start_pos == 0}
    assert lengths == {2, 3}


def test_empty_nullable_rules_zero_weight_and_global_work_budgets():
    from fpcode.contracts.composition import OptionalRule
    for root in (OptionalRule(TerminalMatch('A')), RepeatRule(OptionalRule(TerminalMatch('A')), 1, 2)):
        with pytest.raises(ValueError, match='empty-match|nullable'):
            match_rule_on_sequence(GrammarRule('invalid', root), ('A',))
    rule = GrammarRule('zero', ChoiceRule((TerminalMatch('A'), TerminalMatch('B')), (0., 1.)))
    assert not match_rule_on_sequence(rule, ('A',))
    with pytest.raises(ValueError, match='budget'):
        match_rule_on_sequence(GrammarRule('A', TerminalMatch('A')), ('A',)*20, max_work=2)
    with pytest.raises(ValueError, match='memory budget'):
        match_rule_on_sequence(GrammarRule('A', TerminalMatch('A')), ('A',), memory_budget_bytes=100)


def test_syntactic_match_inside_recording_is_not_observed_completion():
    rule = GrammarRule('AB', SequenceRule((TerminalMatch('A'), TerminalMatch('B'))))
    stream = UnitStream(SESSION, 'primitive', 'lower', 'video', 10., (FrameSpan(0, 20),),
                        (UnitType('A'), UnitType('B'), UnitType('X')),
                        tuple(UnitOccurrence(str(i), t, FrameSpan(5*i, 5*(i+1))) for i, t in enumerate(('X','A','B','X'))))
    result = GrammarEpisodeDetector(GrammarSpecification('g', (rule,))).detect(stream)
    evidence = result.completion[0]
    assert evidence.status == 'unknown' and not evidence.observed_exit
    assert evidence.evidence_kind == 'syntactic_acceptance' and evidence.score is None
    assert result.derivations[0].parent_occurrence_id == result.occurrences[0].occurrence_id
    assert result.derivations[0].grammar_id == 'g'


def test_shared_subtree_depth_cap_and_multi_parent_episode_reuse():
    from fpcode.behavior.compatibility import StreamRequirements, require_compatible
    shared = SequenceRule((TerminalMatch('A'),TerminalMatch('B')))
    root = SequenceRule((shared,SequenceRule((TerminalMatch('A'),shared))))
    with pytest.raises(ValueError,match='depth budget'):
        match_rule_on_sequence(GrammarRule('depth',root),('A','B','A','A','B'),max_depth=2)
    run = TerminalMatch('E_run')
    spec = GrammarSpecification('reuse',(
        GrammarRule('r1',SequenceRule((run,TerminalMatch('E_turn')))),
        GrammarRule('r2',SequenceRule((run,ChoiceRule((TerminalMatch('E_turn'),TerminalMatch('E_rest')))))),
    ))
    catalog = (UnitType('E_run'),UnitType('E_turn'),UnitType('E_rest'))
    source = UnitStream(SESSION,'episode','episode_library','video',10.,(FrameSpan(0,20),),catalog,
                        (UnitOccurrence('run','E_run',FrameSpan(0,10)),UnitOccurrence('turn','E_turn',FrameSpan(10,20))))
    result = GrammarAssemblyComposer(spec).compose(source)
    assert len(result.occurrences) == 2
    assert result.derivations[0].child_references == result.derivations[1].child_references
    assert result.derivations[0].parent_occurrence_id != result.derivations[1].parent_occurrence_id
    assert result.source_streams[0] is source and source.catalog == catalog
    with pytest.raises(ValueError,match='overlapping'):
        require_compatible(result.stream,StreamRequirements(exclusive=True))
