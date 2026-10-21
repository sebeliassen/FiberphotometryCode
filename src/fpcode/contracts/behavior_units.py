"""Method-independent behavioural occurrences on an explicit native-frame clock.

These records describe a detector's available output, not a claim that every
detector identifies equivalent units. Occurrences may overlap and need not tile
valid recording time. Downstream consumers must declare any ordering,
exclusivity or additional capability they require; phases and completion cannot
be inferred from the common interval representation.
"""

import math
from bisect import bisect_right
from dataclasses import dataclass
from numbers import Integral, Real

from .sessions import SessionKey

_LEVELS = frozenset({"primitive", "episode", "assembly", "context"})
_SCORE_KINDS = frozenset({"model_score", "posterior", "probability"})


def _identity(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


@dataclass(frozen=True)
class FrameSpan:
    """Nonempty half-open native-frame interval ``[start, stop)``."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if any(not isinstance(value, Integral) or isinstance(value, bool) for value in (self.start, self.stop)):
            raise ValueError("native-frame bounds must be integers")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("native-frame span requires 0 <= start < stop")
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "stop", int(self.stop))


@dataclass(frozen=True)
class UnitType:
    """One detector-local catalogue entry; IDs are scoped to its model."""

    type_id: str
    description: str = ""

    def __post_init__(self) -> None:
        _identity(self.type_id, "unit type identity")
        if not isinstance(self.description, str):
            raise ValueError("unit description must be a string")


@dataclass(frozen=True)
class UnitOccurrence:
    """An occurrence with optional, explicitly interpreted detector evidence.

    ``model_score`` is a finite detector-specific quantity, not calibrated
    confidence. ``posterior`` and ``probability`` lie in [0, 1], but remain
    interpretable only within the supplying model's declared semantics.
    """

    occurrence_id: str
    type_id: str
    interval: FrameSpan
    score: float | None = None
    score_kind: str | None = None

    def __post_init__(self) -> None:
        _identity(self.occurrence_id, "occurrence identity")
        _identity(self.type_id, "unit type identity")
        if not isinstance(self.interval, FrameSpan):
            raise ValueError("occurrence interval must be a FrameSpan")
        if self.score is None:
            if self.score_kind is not None:
                raise ValueError("score kind requires a score")
            return
        if not isinstance(self.score, Real) or isinstance(self.score, bool) or not math.isfinite(self.score):
            raise ValueError("occurrence score must be finite and numeric")
        if not isinstance(self.score_kind, str) or self.score_kind not in _SCORE_KINDS:
            raise ValueError("score requires explicit model_score, posterior or probability kind")
        if self.score_kind in {"posterior", "probability"} and not 0 <= self.score <= 1:
            raise ValueError("posterior and probability scores must lie between zero and one")
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True)
class UnitStream:
    """Immutable detector output with native-frame validity and stable identity.

    Valid spans must be chronological and disjoint, and each occurrence must
    fit wholly in one span. Touching valid spans are allowed, but an occurrence
    cannot bridge their declared boundary. Occurrences preserve caller order
    and can overlap; consumers choose chronology and overlap policies explicitly.
    An empty stream is valid and retains its catalogue and observation support.
    ``fps`` supplies a constant conversion on ``clock``; it does not synchronise
    different clocks or replace explicit mappings for variable-rate sources.
    """

    session: SessionKey
    level: str
    model_id: str
    clock: str
    fps: float
    valid_intervals: tuple[FrameSpan, ...]
    catalog: tuple[UnitType, ...]
    occurrences: tuple[UnitOccurrence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionKey):
            raise ValueError("stream session must be a SessionKey")
        if not isinstance(self.level, str) or self.level not in _LEVELS:
            raise ValueError("unit level must be primitive, episode, assembly or context")
        _identity(self.model_id, "model identity")
        _identity(self.clock, "native-frame clock identity")
        if not isinstance(self.fps, Real) or isinstance(self.fps, bool) or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        try:
            valid_intervals = tuple(self.valid_intervals)
            catalog = tuple(self.catalog)
            occurrences = tuple(self.occurrences)
        except TypeError as error:
            raise ValueError("valid spans, catalogue and occurrences must be iterable") from error
        if any(not isinstance(span, FrameSpan) for span in valid_intervals):
            raise ValueError("valid intervals must contain FrameSpan records")
        if any(right.start < left.stop for left, right in zip(valid_intervals, valid_intervals[1:])):
            raise ValueError("valid intervals must be ordered and nonoverlapping")
        if any(not isinstance(unit, UnitType) for unit in catalog):
            raise ValueError("catalogue must contain UnitType records")
        type_ids = {unit.type_id for unit in catalog}
        if len(type_ids) != len(catalog):
            raise ValueError("catalogue unit type identities must be unique")
        if any(not isinstance(event, UnitOccurrence) for event in occurrences):
            raise ValueError("occurrences must contain UnitOccurrence records")
        if len({event.occurrence_id for event in occurrences}) != len(occurrences):
            raise ValueError("occurrence identities must be unique")
        starts = tuple(span.start for span in valid_intervals)
        for event in occurrences:
            if event.type_id not in type_ids:
                raise ValueError("occurrence unit type must reference the stream catalogue")
            index = bisect_right(starts, event.interval.start) - 1
            if index < 0 or event.interval.stop > valid_intervals[index].stop:
                raise ValueError("occurrence must fit wholly inside one valid interval")
        object.__setattr__(self, "fps", float(self.fps))
        object.__setattr__(self, "valid_intervals", valid_intervals)
        object.__setattr__(self, "catalog", catalog)
        object.__setattr__(self, "occurrences", occurrences)


__all__ = ["FrameSpan", "UnitType", "UnitOccurrence", "UnitStream"]
