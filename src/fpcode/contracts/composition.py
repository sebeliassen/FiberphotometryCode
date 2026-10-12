"""Finite grammar specification and composition records for behavioural units.

Provides immutable definitions for caller-specified finite grammars (terminals,
sequences, weighted choices, optionals, and bounded repetitions) and their parse
records. Data definitions and validation only; parsing engines belong in statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

from .behavior_results import ScopedOccurrenceReference
from .behavior_units import FrameSpan

_SCORE_KINDS = frozenset({"model_score", "posterior", "probability"})


def _require_nonempty_str(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class TerminalMatch:
    """Matches a specific unit type or any of an interchangeable set."""

    symbol: str
    interchangeable: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_str(self.symbol, "terminal symbol")
        try:
            interchangeable = tuple(self.interchangeable)
        except TypeError as err:
            raise ValueError("interchangeable must be iterable") from err
        for alt in interchangeable:
            _require_nonempty_str(alt, "interchangeable symbol")
        object.__setattr__(self, "interchangeable", interchangeable)

    @property
    def accepted_symbols(self) -> frozenset[str]:
        return frozenset({self.symbol, *self.interchangeable})


@dataclass(frozen=True)
class SequenceRule:
    """Ordered sequence of components that must match consecutively."""

    elements: tuple[RuleComponent, ...]

    def __post_init__(self) -> None:
        try:
            elements = tuple(self.elements)
        except TypeError as err:
            raise ValueError("elements must be iterable") from err
        if len(elements) < 2:
            raise ValueError("sequence rule requires at least two elements")
        for elem in elements:
            if not isinstance(elem, (TerminalMatch, SequenceRule, ChoiceRule, OptionalRule, RepeatRule)):
                raise ValueError(f"invalid sequence element: {type(elem)}")
        object.__setattr__(self, "elements", elements)


@dataclass(frozen=True)
class ChoiceRule:
    """Weighted choice among multiple alternatives."""

    options: tuple[RuleComponent, ...]
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        try:
            options = tuple(self.options)
        except TypeError as err:
            raise ValueError("options must be iterable") from err
        if len(options) < 2:
            raise ValueError("choice rule requires at least two options")
        for opt in options:
            if not isinstance(opt, (TerminalMatch, SequenceRule, ChoiceRule, OptionalRule, RepeatRule)):
                raise ValueError(f"invalid choice option: {type(opt)}")
        object.__setattr__(self, "options", options)

        if self.weights is not None:
            try:
                weights = tuple(float(w) for w in self.weights)
            except (TypeError, ValueError) as err:
                raise ValueError("weights must be iterable of numbers") from err
            if len(weights) != len(options):
                raise ValueError("weights count must match options count")
            if any(w < 0.0 or not math.isfinite(w) for w in weights):
                raise ValueError("weights must be non-negative and finite")
            if sum(weights) <= 0.0:
                raise ValueError("sum of weights must be positive")
            object.__setattr__(self, "weights", weights)


@dataclass(frozen=True)
class OptionalRule:
    """Optional component with inclusion and skip weights."""

    element: RuleComponent
    skip_weight: float = 0.5
    include_weight: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.element, (TerminalMatch, SequenceRule, ChoiceRule, OptionalRule, RepeatRule)):
            raise ValueError(f"invalid optional element: {type(self.element)}")
        for name, w in [("skip_weight", self.skip_weight), ("include_weight", self.include_weight)]:
            if not isinstance(w, Real) or isinstance(w, bool) or not math.isfinite(w) or w < 0:
                raise ValueError(f"{name} must be a non-negative finite number")
        total = float(self.skip_weight) + float(self.include_weight)
        if total <= 0:
            raise ValueError("sum of optional weights must be positive")
        object.__setattr__(self, "skip_weight", float(self.skip_weight))
        object.__setattr__(self, "include_weight", float(self.include_weight))


@dataclass(frozen=True)
class RepeatRule:
    """Bounded repetition of a component with declared repeat limits."""

    element: RuleComponent
    min_repeats: int
    max_repeats: int
    count_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.element, (TerminalMatch, SequenceRule, ChoiceRule, OptionalRule, RepeatRule)):
            raise ValueError(f"invalid repeat element: {type(self.element)}")
        for name, val in [("min_repeats", self.min_repeats), ("max_repeats", self.max_repeats)]:
            if not isinstance(val, Integral) or isinstance(val, bool) or val < 1:
                raise ValueError(f"{name} must be a positive integer >= 1")
        if self.min_repeats > self.max_repeats:
            raise ValueError("min_repeats cannot exceed max_repeats")
        object.__setattr__(self, "min_repeats", int(self.min_repeats))
        object.__setattr__(self, "max_repeats", int(self.max_repeats))

        if self.count_weights is not None:
            expected_len = self.max_repeats - self.min_repeats + 1
            try:
                weights = tuple(float(w) for w in self.count_weights)
            except (TypeError, ValueError) as err:
                raise ValueError("count_weights must be iterable of numbers") from err
            if len(weights) != expected_len:
                raise ValueError(f"count_weights length must be {expected_len}")
            if any(w < 0.0 or not math.isfinite(w) for w in weights):
                raise ValueError("count_weights must be non-negative and finite")
            if sum(weights) <= 0.0:
                raise ValueError("sum of count_weights must be positive")
            object.__setattr__(self, "count_weights", weights)


RuleComponent = TerminalMatch | SequenceRule | ChoiceRule | OptionalRule | RepeatRule


@dataclass(frozen=True)
class GrammarRule:
    """A named grammar rule definition."""

    rule_id: str
    root: RuleComponent
    description: str = ""

    def __post_init__(self) -> None:
        _require_nonempty_str(self.rule_id, "rule identity")
        if not isinstance(self.root, (TerminalMatch, SequenceRule, ChoiceRule, OptionalRule, RepeatRule)):
            raise ValueError(f"invalid rule root: {type(self.root)}")
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")


@dataclass(frozen=True)
class GrammarSpecification:
    """Collection of declared grammar rules with unique identities."""

    grammar_id: str
    rules: tuple[GrammarRule, ...]

    def __post_init__(self) -> None:
        _require_nonempty_str(self.grammar_id, "grammar identity")
        try:
            rules = tuple(self.rules)
        except TypeError as err:
            raise ValueError("rules must be iterable") from err
        if not rules:
            raise ValueError("grammar specification requires at least one rule")
        for rule in rules:
            if not isinstance(rule, GrammarRule):
                raise ValueError("rules must contain GrammarRule instances")
        rule_ids = {r.rule_id for r in rules}
        if len(rule_ids) != len(rules):
            raise ValueError("rule identities must be unique")
        object.__setattr__(self, "rules", rules)


@dataclass(frozen=True)
class ParseMatch:
    """Result of parsing a sequence against a grammar rule."""

    rule_id: str
    derivation_id: str
    start_index: int
    stop_index: int
    matched_span: FrameSpan
    score: float
    score_kind: str = "model_score"
    child_references: tuple[ScopedOccurrenceReference, ...] = ()
    truncated: bool = False

    def __post_init__(self) -> None:
        _require_nonempty_str(self.rule_id, "rule identity")
        _require_nonempty_str(self.derivation_id, "derivation identity")
        if self.start_index < 0 or self.stop_index <= self.start_index:
            raise ValueError("parse indices require 0 <= start_index < stop_index")
        if not isinstance(self.matched_span, FrameSpan):
            raise ValueError("matched span must be a FrameSpan")
        if not isinstance(self.score, Real) or isinstance(self.score, bool) or not math.isfinite(self.score):
            raise ValueError("score must be finite and numeric")
        if self.score_kind not in _SCORE_KINDS:
            raise ValueError(f"score kind must be one of {sorted(_SCORE_KINDS)}")
        if self.score_kind in {"posterior", "probability"} and not 0 <= self.score <= 1:
            raise ValueError("probability score must lie in [0, 1]")
        try:
            children = tuple(self.child_references)
        except TypeError as err:
            raise ValueError("child references must be iterable") from err
        if any(not isinstance(c, ScopedOccurrenceReference) for c in children):
            raise ValueError("child references must contain ScopedOccurrenceReference instances")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "child_references", children)


__all__ = [
    "TerminalMatch",
    "SequenceRule",
    "ChoiceRule",
    "OptionalRule",
    "RepeatRule",
    "RuleComponent",
    "GrammarRule",
    "GrammarSpecification",
    "ParseMatch",
]
