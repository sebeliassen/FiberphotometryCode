"""Inspectable downstream requirements for method-independent unit streams.

Shape compatibility is not scientific equivalence. Different methods may have
different catalogues, event counts, coverage and boundaries. These checks state
which result can enter a consumer and which information is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

from fpcode.contracts.behavior_results import BehaviorResult
from fpcode.contracts.behavior_units import UnitStream
from fpcode.contracts.sessions import SessionKey


@dataclass(frozen=True)
class StreamRequirements:
    """Small consumer contract; no detector selection or inference occurs here."""

    levels: tuple[str, ...] = ("primitive", "episode", "assembly", "context")
    session: SessionKey | None = None
    clock: str | None = None
    fps: float | None = None
    exclusive: bool = False
    scored: bool = False
    uniform_score_kind: bool = False
    required_information: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompatibilityReport:
    compatible: bool
    reasons: tuple[str, ...]


def inspect_compatibility(stream: UnitStream | BehaviorResult, requirements: StreamRequirements = StreamRequirements()) -> CompatibilityReport:
    """Report incompatibilities without coercing clocks, scores or overlaps."""
    underlying_stream: UnitStream
    available: set[str]

    if isinstance(stream, BehaviorResult):
        underlying_stream = stream.stream
        available = stream.available_capabilities()
    elif isinstance(stream, UnitStream):
        underlying_stream = stream
        available = {"intervals", "catalog", "validity", "identity"}
    else:
        return CompatibilityReport(False, ("adapt the upstream output to UnitStream or BehaviorResult first",))

    problems: list[str] = []
    if underlying_stream.level not in requirements.levels:
        problems.append("consumer does not accept unit level " + underlying_stream.level)
    if requirements.session is not None and underlying_stream.session != requirements.session:
        problems.append("recording identity differs; an explicit recording association is required")
    if requirements.clock is not None and underlying_stream.clock != requirements.clock:
        problems.append("clock differs; an explicit clock mapping is required")
    if requirements.fps is not None and underlying_stream.fps != requirements.fps:
        problems.append("fps differs; implicit resampling is not permitted")

    ordered = sorted(underlying_stream.occurrences, key=lambda event: (event.interval.start, event.interval.stop))
    if requirements.exclusive and any(right.interval.start < left.interval.stop for left, right in zip(ordered, ordered[1:])):
        problems.append("overlapping occurrences require an explicit exclusive-view policy")
    if requirements.scored and any(event.score is None for event in ordered):
        problems.append("consumer requires scores for every occurrence")
    if requirements.uniform_score_kind and len({event.score_kind for event in ordered}) > 1:
        problems.append("mixed score kinds cannot be ranked together")

    if isinstance(stream, UnitStream) and ordered and all(event.score is not None for event in ordered):
        available.add("scores")

    for capability in requirements.required_information:
        if capability not in available:
            src_name = "BehaviorResult" if isinstance(stream, BehaviorResult) else "UnitStream"
            problems.append(f"required information is unavailable in {src_name}: {capability}")

    return CompatibilityReport(not problems, tuple(problems))


def require_compatible(stream: UnitStream | BehaviorResult, requirements: StreamRequirements = StreamRequirements()) -> None:
    """Fail before a consumer silently invents unsupported input semantics."""
    report = inspect_compatibility(stream, requirements)
    if not report.compatible:
        raise ValueError("; ".join(report.reasons))


__all__ = ["StreamRequirements", "CompatibilityReport", "inspect_compatibility", "require_compatible"]
