"""Larger assemblies composed from reusable episode definitions.

Composes episode results into higher-order behavioural assemblies using caller-
specified finite grammars. Preserves episode definitions and scoped child
references, supports DAG-like reuse of child types across multiple assemblies,
and maintains strict separation from episode detector implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

from fpcode.behavior.compatibility import StreamRequirements, require_compatible
from fpcode.behavior.unit_adapters import occurrence_blocks
from fpcode.contracts.behavior_results import (
    BehaviorResult,
    CompletionEvidence,
    CompositionDerivation,
    ScopedOccurrenceReference,
)
from fpcode.contracts.behavior_units import FrameSpan, UnitOccurrence, UnitStream, UnitType
from fpcode.contracts.composition import GrammarSpecification
from fpcode.statistics.finite_composition import match_rule_on_sequence

IMPLEMENTATION_STATUS = "implemented"


@dataclass(frozen=True)
class GrammarAssemblyComposer:
    """Assembly composer using caller-specified finite grammar over episode streams."""

    grammar: GrammarSpecification
    model_id: str = "grammar_assemblies"

    def __post_init__(self) -> None:
        if not isinstance(self.grammar, GrammarSpecification):
            raise ValueError("grammar must be a GrammarSpecification")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a nonempty string")

    def compose(
        self,
        stream: UnitStream,
        *,
        max_derivations: int = 50,
        max_candidates: int = 10000,
        **search_limits,
    ) -> BehaviorResult:
        """Compose episode stream into assembly occurrences with derivation sidecars."""
        require_compatible(stream, StreamRequirements(levels=("episode",), exclusive=True))
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1:
            raise ValueError("max_candidates must be a positive integer")
        if len(self.grammar.rules) > 128:
            raise ValueError("first-pass grammar wrappers support at most 128 declared rules")

        blocks = occurrence_blocks(stream)

        assembly_occurrences: list[UnitOccurrence] = []
        derivations: list[CompositionDerivation] = []
        completions: list[CompletionEvidence] = []
        as_counter = 0

        for block in blocks:
            tokens = tuple(occ.type_id for occ in block)
            for rule in self.grammar.rules:
                matches = match_rule_on_sequence(rule, tokens, max_derivations=max_derivations, **search_limits)
                for m in matches:
                    if len(assembly_occurrences) >= max_candidates:
                        raise ValueError("grammar candidate budget exceeded; search is not exhaustive")
                    start_frame = block[m.start_pos].interval.start
                    stop_frame = block[m.stop_pos - 1].interval.stop
                    span = FrameSpan(start_frame, stop_frame)
                    occ_id = f"assembly_{as_counter}"
                    as_counter += 1

                    child_refs = tuple(
                        ScopedOccurrenceReference(
                            session=stream.session,
                            model_id=stream.model_id,
                            occurrence_id=block[idx].occurrence_id,
                        )
                        for idx in m.child_token_indices
                    )

                    score_kind = "model_score"  # grammar log path weight, not completion confidence
                    assembly_occurrences.append(
                        UnitOccurrence(
                            occurrence_id=occ_id,
                            type_id=rule.rule_id,
                            interval=span,
                            score=m.log_score,
                            score_kind=score_kind,
                        )
                    )

                    derivations.append(
                        CompositionDerivation(
                            rule_id=rule.rule_id,
                            derivation_id=f"{occ_id}_{m.derivation_id}",
                            child_references=child_refs,
                            score=m.log_score,
                            score_kind=score_kind,
                            truncated=m.truncated,
                            interval=span,
                            parent_occurrence_id=occ_id,
                            trace=m.trace,
                            grammar_id=self.grammar.grammar_id,
                        )
                    )

                    at_end_of_block = (m.stop_pos >= len(block))
                    at_end_of_valid = (stop_frame >= block[-1].interval.stop)
                    at_start = m.start_pos == 0
                    at_end = at_end_of_block or at_end_of_valid

                    completions.append(
                        CompletionEvidence(
                            occurrence_id=occ_id,
                            observed_entry=False,
                            observed_exit=False,
                            final_phase_reached=False,
                            status="censored" if at_start or at_end else "unknown",
                            left_censored=at_start,
                            right_censored=at_end,
                            evidence_kind="syntactic_acceptance",
                        )
                    )

        catalog = tuple(UnitType(r.rule_id, r.description) for r in self.grammar.rules)
        as_stream = UnitStream(
            session=stream.session,
            level="assembly",
            model_id=self.model_id,
            clock=stream.clock,
            fps=stream.fps,
            valid_intervals=stream.valid_intervals,
            catalog=catalog,
            occurrences=tuple(assembly_occurrences),
        )

        return BehaviorResult(
            stream=as_stream,
            derivations=tuple(derivations),
            completion=tuple(completions),
            source_streams=(stream,),
            diagnostics={"grammar_id": self.grammar.grammar_id,
                         "pruned": any(d.truncated for d in derivations),
                         "score_semantics": "grammar_log_path_weight"},
        )


__all__ = [
    "IMPLEMENTATION_STATUS",
    "GrammarAssemblyComposer",
]
