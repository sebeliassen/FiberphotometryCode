"""Validated result envelopes and optional rich payloads for behavioural units.

UnitStream remains the method-independent interval representation. This module
defines optional validated sidecars (features, phases, completion evidence,
composition derivations, and posterior distributions) and a unified result
envelope that preserves compatibility while exposing rich capabilities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from .behavior_units import FrameSpan, UnitOccurrence, UnitStream, UnitType
from .sessions import SessionKey
from .signals import SignalSeries
from .timelines import frozen_array

_COMPLETION_STATUSES = frozenset({"completed", "deviated", "censored", "unknown"})
_SCORE_KINDS = frozenset({"model_score", "posterior", "probability"})
_OBSERVATION_COORDINATES = frozenset({"event", "frame"})


def _require_nonempty_str(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _freeze_diagnostic(value):
    """Own metadata recursively; fitted instances are not shared result payloads."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("diagnostic mapping keys must be strings")
        return MappingProxyType({key: _freeze_diagnostic(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_diagnostic(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufcSU":
            raise ValueError("diagnostic arrays must contain numbers or strings")
        return frozen_array(value)
    if value is None or isinstance(value, (str, Real)):
        return value
    raise ValueError("diagnostics must contain data, not fitted objects or callbacks")


@dataclass(frozen=True)
class ScopedOccurrenceReference:
    """Globally scoped reference to a specific unit occurrence."""

    session: SessionKey
    model_id: str
    occurrence_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionKey):
            raise ValueError("reference session must be a SessionKey")
        _require_nonempty_str(self.model_id, "model identity")
        _require_nonempty_str(self.occurrence_id, "occurrence identity")


@dataclass(frozen=True)
class FeatureAssociation:
    """Associated continuous features with recording and coordinate-frame provenance."""

    series: SignalSeries
    session: SessionKey
    coordinate_frame: str
    provenance: str = ""
    normalized_series: SignalSeries | None = None
    native_frames: np.ndarray | None = None
    sample_exposure_s: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.series, SignalSeries):
            raise ValueError("features must wrap a SignalSeries")
        if not isinstance(self.session, SessionKey):
            raise ValueError("session must be a SessionKey")
        _require_nonempty_str(self.coordinate_frame, "coordinate frame")
        if not isinstance(self.provenance, str):
            raise ValueError("provenance must be a string")
        if self.normalized_series is not None:
            if not isinstance(self.normalized_series, SignalSeries):
                raise ValueError("normalized series must be a SignalSeries")
            if len(self.normalized_series.time) != len(self.series.time):
                raise ValueError("normalized series length must match raw series")
            normalized = self.normalized_series
            if (normalized.time.clock != self.series.time.clock
                    or normalized.time.unit != self.series.time.unit
                    or not np.array_equal(normalized.time.values, self.series.time.values)
                    or tuple(s.name for s in normalized.signals) != tuple(s.name for s in self.series.signals)
                    or normalized.values.shape != self.series.values.shape
                    or np.any(normalized.valid & ~self.series.valid)):
                raise ValueError("normalized series must preserve raw coordinates, channels and support")
        if self.native_frames is not None:
            frames = frozen_array(self.native_frames)
            if (frames.ndim != 1 or len(frames) != len(self.series.time)
                    or not np.issubdtype(frames.dtype, np.number)
                    or np.any(~np.isfinite(frames)) or np.any(frames < 0)
                    or np.any(frames != np.floor(frames)) or np.any(frames[1:] <= frames[:-1])):
                raise ValueError("native frames must be nonnegative increasing integers, one per sample")
            if self.series.time.unit == "frame" and not np.array_equal(frames, self.series.time.values):
                raise ValueError("native mapping disagrees with frame coordinates")
            object.__setattr__(self, "native_frames", frames)
        if self.sample_exposure_s is not None:
            exposure = frozen_array(self.sample_exposure_s, dtype=float)
            if (exposure.shape != (len(self.series.time),) or np.any(~np.isfinite(exposure))
                    or np.any(exposure < 0)):
                raise ValueError("sample exposure must be a finite nonnegative vector matching samples")
            object.__setattr__(self, "sample_exposure_s", exposure)


@dataclass(frozen=True)
class PhaseRecord:
    """Sub-unit phase or constituent interval belonging to a parent occurrence."""

    parent_occurrence_id: str
    phase_id: str
    order: int
    interval: FrameSpan
    source_reference: ScopedOccurrenceReference | None = None
    evidence_kind: str = "model"

    def __post_init__(self) -> None:
        _require_nonempty_str(self.parent_occurrence_id, "parent occurrence identity")
        _require_nonempty_str(self.phase_id, "phase identity")
        if not isinstance(self.order, Integral) or isinstance(self.order, bool) or self.order < 0:
            raise ValueError("phase order must be a non-negative integer")
        if not isinstance(self.interval, FrameSpan):
            raise ValueError("phase interval must be a FrameSpan")
        if self.source_reference is not None and not isinstance(self.source_reference, ScopedOccurrenceReference):
            raise ValueError("source reference must be a ScopedOccurrenceReference")
        _require_nonempty_str(self.evidence_kind, "evidence kind")
        object.__setattr__(self, "order", int(self.order))


@dataclass(frozen=True)
class CompletionEvidence:
    """Boundary, progression and termination evidence for a unit occurrence."""

    occurrence_id: str
    observed_entry: bool = True
    observed_exit: bool = True
    final_phase_reached: bool = False
    status: str = "unknown"
    left_censored: bool = False
    right_censored: bool = False
    score: float | None = None
    score_kind: str | None = None
    evidence_kind: str = "declared"

    def __post_init__(self) -> None:
        _require_nonempty_str(self.occurrence_id, "occurrence identity")
        for bool_field, val in [
            ("observed_entry", self.observed_entry),
            ("observed_exit", self.observed_exit),
            ("final_phase_reached", self.final_phase_reached),
            ("left_censored", self.left_censored),
            ("right_censored", self.right_censored),
        ]:
            if not isinstance(val, bool):
                raise ValueError(f"{bool_field} must be a boolean")
        if self.status not in _COMPLETION_STATUSES:
            raise ValueError(f"status must be one of {sorted(_COMPLETION_STATUSES)}")
        _require_nonempty_str(self.evidence_kind, "completion evidence kind")
        if self.right_censored and self.observed_exit:
            raise ValueError("right-censored evidence cannot have an observed exit")
        if self.left_censored and self.observed_entry:
            raise ValueError("left-censored evidence cannot have an observed entry")
        if self.status in {"completed", "deviated"} and (not self.observed_exit or self.right_censored):
            raise ValueError("evaluable completion requires an observed exit")
        if self.score is not None:
            if not isinstance(self.score, Real) or isinstance(self.score, bool) or not math.isfinite(self.score):
                raise ValueError("score must be finite and numeric")
            if self.score_kind not in _SCORE_KINDS:
                raise ValueError(f"score kind must be one of {sorted(_SCORE_KINDS)}")
            if self.score_kind in {"posterior", "probability"} and not 0 <= self.score <= 1:
                raise ValueError("probability score must lie in [0, 1]")
            object.__setattr__(self, "score", float(self.score))
        elif self.score_kind is not None:
            raise ValueError("score kind requires a score")


@dataclass(frozen=True)
class CompositionDerivation:
    """Derivation evidence for a composed rule match with child references."""

    rule_id: str
    derivation_id: str
    child_references: tuple[ScopedOccurrenceReference, ...]
    score: float
    score_kind: str = "model_score"
    truncated: bool = False
    interval: FrameSpan | None = None
    parent_occurrence_id: str | None = None
    trace: tuple[str, ...] = ()
    grammar_id: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.rule_id, "rule identity")
        _require_nonempty_str(self.derivation_id, "derivation identity")
        try:
            children = tuple(self.child_references)
        except TypeError as err:
            raise ValueError("child references must be iterable") from err
        if any(not isinstance(c, ScopedOccurrenceReference) for c in children):
            raise ValueError("child references must contain ScopedOccurrenceReference")
        if not isinstance(self.score, Real) or isinstance(self.score, bool) or not math.isfinite(self.score):
            raise ValueError("derivation score must be finite and numeric")
        if self.score_kind not in _SCORE_KINDS:
            raise ValueError(f"score kind must be one of {sorted(_SCORE_KINDS)}")
        if self.score_kind in {"posterior", "probability"} and not 0 <= self.score <= 1:
            raise ValueError("probability score must lie in [0, 1]")
        if not isinstance(self.truncated, bool):
            raise ValueError("truncated flag must be a boolean")
        if self.interval is not None and not isinstance(self.interval, FrameSpan):
            raise ValueError("interval must be a FrameSpan")
        if self.parent_occurrence_id is not None:
            _require_nonempty_str(self.parent_occurrence_id, "derivation parent identity")
        if self.grammar_id is not None:
            _require_nonempty_str(self.grammar_id, "grammar identity")
        trace = tuple(self.trace)
        if any(not isinstance(x, str) for x in trace):
            raise ValueError("derivation trace must contain strings")
        object.__setattr__(self, "trace", trace)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "child_references", children)


@dataclass(frozen=True)
class StatePosteriorRecord:
    """Validated posterior probabilities over a state catalogue."""

    observation_coordinate: str
    state_catalog: tuple[UnitType, ...]
    probabilities: np.ndarray
    valid_mask: np.ndarray
    sample_intervals: tuple[FrameSpan, ...]
    session: SessionKey
    model_id: str
    clock: str | None = None

    def __post_init__(self) -> None:
        if self.observation_coordinate not in _OBSERVATION_COORDINATES:
            raise ValueError(f"observation coordinate must be one of {sorted(_OBSERVATION_COORDINATES)}")
        if not isinstance(self.session, SessionKey):
            raise ValueError("session must be a SessionKey")
        _require_nonempty_str(self.model_id, "model identity")
        if self.clock is not None:
            _require_nonempty_str(self.clock, "posterior clock")
        try:
            catalog = tuple(self.state_catalog)
        except TypeError as err:
            raise ValueError("catalog must be iterable") from err
        if any(not isinstance(u, UnitType) for u in catalog):
            raise ValueError("catalog must contain UnitType instances")
        n_states = len(catalog)
        if n_states == 0:
            raise ValueError("state catalog cannot be empty")
        if len({u.type_id for u in catalog}) != n_states:
            raise ValueError("catalog type identities must be unique")

        probs = frozen_array(self.probabilities, dtype=float)
        valid = frozen_array(self.valid_mask, dtype=bool)
        if probs.ndim != 2 or probs.shape[1] != n_states:
            raise ValueError("probabilities array must have shape (observations, n_states)")
        if valid.ndim != 1 or valid.shape[0] != probs.shape[0]:
            raise ValueError("valid mask must have shape (observations,)")
        if np.any(valid[:, None] & ((probs < 0.0) | (probs > 1.0) | ~np.isfinite(probs))):
            raise ValueError("supported probabilities must lie in [0, 1]")
        if np.any(valid):
            row_sums = probs[valid].sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-4):
                raise ValueError("supported probability rows must sum to 1")

        try:
            intervals = tuple(self.sample_intervals)
        except TypeError as err:
            raise ValueError("sample intervals must be iterable") from err
        if len(intervals) != probs.shape[0]:
            raise ValueError("sample intervals length must match observation count")
        if any(not isinstance(span, FrameSpan) for span in intervals):
            raise ValueError("sample intervals must contain FrameSpan instances")

        object.__setattr__(self, "state_catalog", catalog)
        object.__setattr__(self, "probabilities", probs)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "sample_intervals", intervals)


@dataclass(frozen=True)
class BehaviorResult:
    """Unified result envelope pairing a UnitStream with optional validated payloads."""

    stream: UnitStream
    features: FeatureAssociation | None = None
    phases: tuple[PhaseRecord, ...] | None = None
    completion: tuple[CompletionEvidence, ...] | None = None
    derivations: tuple[CompositionDerivation, ...] | None = None
    posteriors: StatePosteriorRecord | None = None
    diagnostics: Mapping | None = None
    source_streams: tuple[UnitStream, ...] = ()
    phase_coverage: str = "ordered"

    def __post_init__(self) -> None:
        if not isinstance(self.stream, UnitStream):
            raise ValueError("result must contain a valid UnitStream")
        occ_ids = {occ.occurrence_id for occ in self.stream.occurrences}
        if self.phase_coverage not in {"ordered", "alternatives"}:
            raise ValueError("phase coverage must be ordered or alternatives")
        sources = tuple(self.source_streams)
        source_index = {(self.stream.model_id, o.occurrence_id): o for o in self.stream.occurrences}
        source_models = {self.stream.model_id}
        for source in sources:
            if not isinstance(source, UnitStream):
                raise ValueError("source streams must be UnitStream records")
            if (source.session != self.stream.session or source.clock != self.stream.clock
                    or source.fps != self.stream.fps):
                raise ValueError("source recording/clock/fps must match result")
            if source.model_id in source_models:
                raise ValueError("source model identities must be unique")
            source_models.add(source.model_id)
            source_index.update({(source.model_id, o.occurrence_id): o for o in source.occurrences})
        object.__setattr__(self, "source_streams", sources)

        def validate_reference(ref, enclosing):
            if ref.session != self.stream.session:
                raise ValueError("source reference session does not match stream session")
            child = source_index.get((ref.model_id, ref.occurrence_id))
            if child is None:
                raise ValueError("source reference must resolve in an associated source stream")
            if child.interval.start < enclosing.start or child.interval.stop > enclosing.stop:
                raise ValueError("source child must be contained in its parent interval")

        if self.features is not None:
            if not isinstance(self.features, FeatureAssociation):
                raise ValueError("features must be a FeatureAssociation")
            if self.features.session != self.stream.session:
                raise ValueError("feature recording does not match stream session")
            if self.features.series.time.clock != self.stream.clock:
                raise ValueError("feature clock does not match stream clock")

        occ_by_id = {occ.occurrence_id: occ for occ in self.stream.occurrences}
        if self.phases is not None:
            try:
                phases = tuple(self.phases)
            except TypeError as err:
                raise ValueError("phases must be iterable") from err
            orders = set()
            by_parent = {}
            for phase in phases:
                if not isinstance(phase, PhaseRecord):
                    raise ValueError("phases must contain PhaseRecord instances")
                if phase.parent_occurrence_id not in occ_by_id:
                    raise ValueError(f"phase parent {phase.parent_occurrence_id} not in stream occurrences")
                parent = occ_by_id[phase.parent_occurrence_id]
                if phase.interval.start < parent.interval.start or phase.interval.stop > parent.interval.stop:
                    raise ValueError(
                        f"phase interval {phase.interval} must be wholly contained within parent interval {parent.interval}"
                    )
                key = (phase.parent_occurrence_id, phase.order)
                if key in orders:
                    raise ValueError("phase orders must be unique within a parent")
                orders.add(key)
                by_parent.setdefault(phase.parent_occurrence_id, []).append(phase)
                if phase.source_reference is not None:
                    validate_reference(phase.source_reference, phase.interval)
            if self.phase_coverage == "ordered":
                for group in by_parent.values():
                    group = sorted(group, key=lambda p: p.order)
                    if any(a.interval.stop > b.interval.start for a, b in zip(group, group[1:])):
                        raise ValueError("ordered phases must progress without overlapping")
            object.__setattr__(self, "phases", phases)

        if self.completion is not None:
            try:
                completion = tuple(self.completion)
            except TypeError as err:
                raise ValueError("completion must be iterable") from err
            comp_ids = set()
            for comp in completion:
                if not isinstance(comp, CompletionEvidence):
                    raise ValueError("completion must contain CompletionEvidence instances")
                if comp.occurrence_id not in occ_ids:
                    raise ValueError(f"completion occurrence {comp.occurrence_id} not in stream occurrences")
                if comp.occurrence_id in comp_ids:
                    raise ValueError(f"duplicate completion evidence for {comp.occurrence_id}")
                comp_ids.add(comp.occurrence_id)
            object.__setattr__(self, "completion", completion)

        if self.derivations is not None:
            try:
                derivs = tuple(self.derivations)
            except TypeError as err:
                raise ValueError("derivations must be iterable") from err
            ids = set()
            for deriv in derivs:
                if not isinstance(deriv, CompositionDerivation):
                    raise ValueError("derivations must contain CompositionDerivation instances")
                if deriv.derivation_id in ids:
                    raise ValueError("derivation identities must be unique")
                ids.add(deriv.derivation_id)
                parent_id = deriv.parent_occurrence_id
                if parent_id is None:
                    candidates = [o for o in self.stream.occurrences if o.interval == deriv.interval]
                    if len(candidates) != 1:
                        raise ValueError("derivation requires an unambiguous parent occurrence identity")
                    parent_id = candidates[0].occurrence_id
                if parent_id not in occ_ids:
                    raise ValueError("derivation parent not in stream occurrences")
                parent = occ_by_id[parent_id]
                if deriv.interval is not None and deriv.interval != parent.interval:
                    raise ValueError("derivation interval must match parent occurrence")
                if len(set(deriv.child_references)) != len(deriv.child_references):
                    raise ValueError("a derivation cannot duplicate a source occurrence")
                for ref in deriv.child_references:
                    validate_reference(ref, parent.interval)
            object.__setattr__(self, "derivations", derivs)

        if self.posteriors is not None:
            if not isinstance(self.posteriors, StatePosteriorRecord):
                raise ValueError("posteriors must be a StatePosteriorRecord")
            if self.posteriors.session != self.stream.session:
                raise ValueError("posterior session does not match stream session")
            if self.posteriors.model_id != self.stream.model_id or self.posteriors.clock != self.stream.clock:
                raise ValueError("posterior model/clock does not match stream")
            if self.stream.level == "context" and self.posteriors.state_catalog != self.stream.catalog:
                raise ValueError("posterior state catalog does not match context catalog")
            for valid, span in zip(self.posteriors.valid_mask, self.posteriors.sample_intervals):
                if valid and not any(v.start <= span.start and span.stop <= v.stop for v in self.stream.valid_intervals):
                    raise ValueError("supported posterior interval is outside stream validity")
            intervals = self.posteriors.sample_intervals
            if any(a.stop > b.start for a, b in zip(intervals, intervals[1:])):
                raise ValueError("posterior observation intervals must be chronological and exclusive")
        if self.diagnostics is not None:
            if not isinstance(self.diagnostics, Mapping):
                raise ValueError("diagnostics must be a mapping")
            object.__setattr__(self, "diagnostics", _freeze_diagnostic(self.diagnostics))

    @property
    def session(self) -> SessionKey:
        return self.stream.session

    @property
    def level(self) -> str:
        return self.stream.level

    @property
    def model_id(self) -> str:
        return self.stream.model_id

    @property
    def clock(self) -> str:
        return self.stream.clock

    @property
    def fps(self) -> float:
        return self.stream.fps

    @property
    def valid_intervals(self) -> tuple[FrameSpan, ...]:
        return self.stream.valid_intervals

    @property
    def catalog(self) -> tuple[UnitType, ...]:
        return self.stream.catalog

    @property
    def occurrences(self) -> tuple[UnitOccurrence, ...]:
        return self.stream.occurrences

    def available_capabilities(self) -> set[str]:
        """Compute capabilities supported by this envelope."""
        caps = {"intervals", "catalog", "validity", "identity"}
        if self.stream.occurrences and all(e.score is not None for e in self.stream.occurrences):
            caps.add("scores")
        if self.features is not None:
            caps.add("features")
        if self.phases is not None:
            caps.add("phases")
        if self.completion is not None:
            caps.add("completion")
        if self.derivations is not None:
            caps.add("derivations")
        if self.posteriors is not None:
            caps.add("posteriors")
        return caps


__all__ = [
    "ScopedOccurrenceReference",
    "FeatureAssociation",
    "PhaseRecord",
    "CompletionEvidence",
    "CompositionDerivation",
    "StatePosteriorRecord",
    "BehaviorResult",
]
