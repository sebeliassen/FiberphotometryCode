"""Memoized finite composition with explicit per-span pruning and global budgets.

Weighted derivations are model-path weights, not behavioural confidence. The
parser retains different matched lengths so local pruning cannot erase the only
length that permits a later sequence component. Budget exhaustion is an explicit
error, never an unreported partial search.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from numbers import Integral

from fpcode.contracts.composition import (
    ChoiceRule,
    GrammarRule,
    OptionalRule,
    RepeatRule,
    RuleComponent,
    SequenceRule,
    TerminalMatch,
)


@dataclass(frozen=True)
class RawDerivation:
    length: int
    score: float
    child_indices: tuple[int, ...]
    truncated: bool = False
    log_score: float = 0.
    trace: tuple[str, ...] = ()


@dataclass(frozen=True)
class GrammarParseResult:
    rule_id: str
    derivation_id: str
    start_pos: int
    stop_pos: int
    score: float
    child_token_indices: tuple[int, ...]
    truncated: bool = False
    log_score: float = 0.
    trace: tuple[str, ...] = ()


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _log_weights(weights):
    scale = max(weights)
    scaled = [w/scale for w in weights]
    total = sum(scaled)
    return [math.log(w)-math.log(total) if w else -math.inf for w in scaled]


class _Parser:
    def __init__(self, root, tokens, *, max_derivations=50, max_work=100000,
                 max_nodes=512, max_depth=64, max_window=256, max_repeats=64,
                 memory_budget_bytes=8*1024*1024, max_matches=10000):
        for name, value in locals().copy().items():
            if name.startswith('max_') or name == 'memory_budget_bytes':
                setattr(self, name, _positive_int(value, name))
        self.tokens = tuple(tokens)
        if any(not isinstance(t, str) or not t for t in self.tokens):
            raise ValueError("tokens must be nonempty strings")
        self.cache = {}
        self.work = self.bytes_used = 0
        self.visited = set()
        self.active = set()
        self.nullable = {}
        self.root_nullable = self._validate(root, 0)

    def _validate(self, node, depth):
        if depth > self.max_depth:
            raise ValueError("grammar depth budget exceeded")
        key = id(node)
        if key in self.active:
            raise ValueError("recursive grammar rules are unsupported")
        if key in self.nullable:
            nullable, height = self.nullable[key]
            if depth + height > self.max_depth:
                raise ValueError("grammar depth budget exceeded")
            return nullable
        self.visited.add(key)
        if len(self.visited) > self.max_nodes:
            raise ValueError("grammar node budget exceeded")
        self.active.add(key)
        if isinstance(node, TerminalMatch):
            nullable = False
            height = 0
        elif isinstance(node, SequenceRule):
            sub = [self._validate_sub(c, depth+1) for c in node.elements]
            nullable = all(n for n, _ in sub)
            height = 1 + max((h for _, h in sub), default=0)
        elif isinstance(node, ChoiceRule):
            sub = [self._validate_sub(c, depth+1) for c in node.options]
            weights = node.weights or (1.,)*len(sub)
            nullable = any(n and w > 0 for (n, _), w in zip(sub, weights))
            height = 1 + max((h for _, h in sub), default=0)
        elif isinstance(node, OptionalRule):
            n, h = self._validate_sub(node.element, depth+1)
            nullable = n or node.skip_weight > 0
            height = 1 + h
        elif isinstance(node, RepeatRule):
            if node.max_repeats > self.max_repeats:
                raise ValueError("grammar repeat budget exceeded")
            n, h = self._validate_sub(node.element, depth+1)
            if n:
                raise ValueError("nullable repeated components are unsupported")
            nullable = False
            height = 1 + h
        else:
            raise ValueError("unsupported grammar component")
        self.active.remove(key)
        self.nullable[key] = (nullable, height)
        return nullable

    def _validate_sub(self, node, depth):
        self._validate(node, depth)
        return self.nullable[id(node)]

    def _charge(self, children=(), trace=()):
        self.work += 1
        # Conservative cumulative allocation budget also bounds intermediate queues.
        self.bytes_used += 160 + 16*len(children) + 64*len(trace)
        if self.work > self.max_work:
            raise ValueError("grammar work budget exceeded; search is not exhaustive")
        if self.bytes_used > self.memory_budget_bytes:
            raise ValueError("grammar memory budget exceeded; search is not exhaustive")

    def _new(self, length, log_score, children=(), truncated=False, trace=()):
        self._charge(children, trace)
        return RawDerivation(length, math.exp(log_score), children, truncated, log_score, trace)

    def _retain(self, candidates):
        groups = defaultdict(list)
        lost = set()
        for d in candidates:
            if d.length > self.max_window:
                raise ValueError("grammar match window budget exceeded")
            group = groups[d.length]
            group.append(d)
            group.sort(key=lambda x: x.log_score, reverse=True)
            if len(group) > self.max_derivations:
                group.pop()
                lost.add(d.length)
        return tuple(replace(d, truncated=True) if length in lost else d
                     for length in sorted(groups) for d in groups[length])

    def parse(self, node, pos):
        key = (id(node), pos)
        if key in self.cache:
            return self.cache[key]
        self._charge()
        if pos > len(self.tokens):
            return ()
        if isinstance(node, TerminalMatch):
            result = (self._new(1, 0., (pos,), trace=(f"terminal:{self.tokens[pos]}",)),) if pos < len(self.tokens) and self.tokens[pos] in node.accepted_symbols else ()
        elif isinstance(node, ChoiceRule):
            def choices():
                weights = _log_weights(node.weights or (1.,)*len(node.options))
                for i, (child, w) in enumerate(zip(node.options, weights)):
                    if not math.isfinite(w):
                        continue
                    for d in self.parse(child, pos):
                        yield self._new(d.length, d.log_score+w, d.child_indices, d.truncated,
                                        (f"choice:{i}",)+d.trace)
            result = self._retain(choices())
        elif isinstance(node, OptionalRule):
            def optional():
                skip, include = _log_weights((node.skip_weight, node.include_weight))
                if math.isfinite(skip):
                    yield self._new(0, skip, trace=("optional:skip",))
                if math.isfinite(include):
                    for d in self.parse(node.element, pos):
                        yield self._new(d.length, include+d.log_score, d.child_indices,
                                        d.truncated, ("optional:include",)+d.trace)
            result = self._retain(optional())
        elif isinstance(node, SequenceRule):
            result = (self._new(0, 0.),)
            for i, child in enumerate(node.elements):
                def combined(previous=result, component=child, index=i):
                    for d in previous:
                        for e in self.parse(component, pos+d.length):
                            yield self._new(d.length+e.length, d.log_score+e.log_score,
                                            d.child_indices+e.child_indices, d.truncated or e.truncated,
                                            d.trace+(f"sequence:{index}",)+e.trace)
                result = self._retain(combined())
                if not result:
                    break
        else:  # validated RepeatRule
            def repeated():
                weights = _log_weights(node.count_weights or (1.,)*(node.max_repeats-node.min_repeats+1))
                layer = (self._new(0, 0.),)
                for count in range(1, node.max_repeats+1):
                    def extended(previous=layer):
                        for d in previous:
                            for e in self.parse(node.element, pos+d.length):
                                yield self._new(d.length+e.length, d.log_score+e.log_score,
                                                d.child_indices+e.child_indices, d.truncated or e.truncated,
                                                d.trace+e.trace)
                    layer = self._retain(extended())
                    if not layer:
                        break
                    if count >= node.min_repeats:
                        w = weights[count-node.min_repeats]
                        if math.isfinite(w):
                            for d in layer:
                                yield self._new(d.length, d.log_score+w, d.child_indices,
                                                d.truncated, (f"repeat:{count}",)+d.trace)
            result = self._retain(repeated())
        self.cache[key] = result
        return result


def parse_component(component: RuleComponent, tokens: tuple[str, ...], start_pos: int,
                    *, max_derivations=50, **limits) -> tuple[RawDerivation, ...]:
    """Parse a component; max_derivations bounds alternatives per matched length."""
    if isinstance(start_pos, bool) or not isinstance(start_pos, Integral) or start_pos < 0:
        raise ValueError("start_pos must be a nonnegative integer")
    parser = _Parser(component, tokens, max_derivations=max_derivations, **limits)
    return parser.parse(component, int(start_pos))


def match_rule_on_sequence(rule: GrammarRule, tokens: tuple[str, ...], *,
                           max_derivations=50, **limits) -> tuple[GrammarParseResult, ...]:
    """Find finite matches; pruning is per span, hard budget failures are explicit."""
    parser = _Parser(rule.root, tokens, max_derivations=max_derivations, **limits)
    if parser.root_nullable:
        raise ValueError("an empty-match grammar root is unsupported")
    results = []
    for pos in range(len(parser.tokens)):
        for i, d in enumerate(parser.parse(rule.root, pos)):
            if d.length:
                parser._charge(d.child_indices, d.trace)
                results.append(GrammarParseResult(rule.rule_id, f"{rule.rule_id}_pos{pos}_d{i}",
                    pos, pos+d.length, d.score, d.child_indices, d.truncated, d.log_score, d.trace))
                if len(results) > parser.max_matches:
                    raise ValueError("grammar match budget exceeded; search is not exhaustive")
    return tuple(results)


__all__ = ["RawDerivation", "GrammarParseResult", "parse_component", "match_rule_on_sequence"]
