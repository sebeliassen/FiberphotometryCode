"""Explicit representation adapters into the shared native-frame unit contract.

These adapters do not fit models, infer confidence, suppress overlapping calls,
or change a caller's vocabulary. Keep the original method output alongside the
common stream; model-specific templates/posteriors are not encoded by intervals.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from fpcode.behavior.primitives.bouts import run_boundaries
from fpcode.contracts.behavior_units import FrameSpan, UnitOccurrence, UnitStream, UnitType
from fpcode.contracts.sessions import SessionKey


def _native_vector(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer vector")
    return values


def from_frame_labels(frames, labels, *, session: SessionKey, model_id: str, clock: str, fps: float,
                      catalog: Iterable[UnitType] | None = None) -> UnitStream:
    """Convert an existing label assignment to primitive occurrences, gap-aware.

    Negative labels remain invalid boundaries. Dropped native frames also split
    validity and occurrences. Catalogue IDs are the original integer labels as
    strings; supplied entries may include absent types. No resolution is chosen.
    """
    frames = _native_vector(frames, "frames")
    labels = _native_vector(labels, "labels")
    if len(frames) != len(labels) or np.any(frames < 0) or np.any(frames[1:] <= frames[:-1]):
        raise ValueError("frames must be nonnegative, strictly increasing and aligned with labels")
    valid = labels >= 0
    starts, stops = run_boundaries(valid, frame_indices=frames)
    support = tuple(FrameSpan(int(frames[a]), int(frames[b - 1]) + 1)
                    for a, b in zip(starts, stops) if valid[a])
    starts, stops = run_boundaries(labels, frame_indices=frames)
    occurrences = tuple(UnitOccurrence(f"bout_{i}", str(int(labels[a])), FrameSpan(int(frames[a]), int(frames[b - 1]) + 1))
                        for i, (a, b) in enumerate(zip(starts, stops)) if valid[a])
    if catalog is None:
        catalog = tuple(UnitType(str(int(label))) for label in np.unique(labels[valid]))
    return UnitStream(session, "primitive", model_id, clock, fps, support, tuple(catalog), occurrences)


def from_occurrence_table(table: pd.DataFrame, *, session: SessionKey, level: str, model_id: str,
                          clock: str, fps: float, valid_intervals: Iterable[FrameSpan], catalog: Iterable[UnitType],
                          type_column="chain_id", start_column="start_frame", stop_column="end_frame_exclusive",
                          id_column=None, score_column=None, score_kind=None) -> UnitStream:
    """Adapt physical-interval tables without guessing their column/score semantics.

    Caller supplies recording identity and validity; a multi-recording table must
    be partitioned first. Bounds are passed to FrameSpan without integer coercion,
    so fractional coordinates cannot quietly become native frames.
    """
    required = [type_column, start_column, stop_column]
    if id_column is not None:
        required.append(id_column)
    if score_column is not None:
        required.append(score_column)
        if score_kind is None:
            raise ValueError("score_column requires explicit score_kind")
    elif score_kind is not None:
        raise ValueError("score_kind requires score_column")
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError("occurrence table missing columns: " + ", ".join(sorted(missing)))
    if "session_id" in table and not table.session_id.eq(session.recording).all():
        raise ValueError("partition occurrences by recording before adapting them")
    occurrences = []
    # Column access avoids pandas coercing mixed numerical rows into float bounds.
    for i in range(len(table)):
        type_id = table[type_column].iloc[i]
        if pd.isna(type_id):
            raise ValueError("occurrence type identity cannot be missing")
        occurrence_id = str(table[id_column].iloc[i]) if id_column is not None else f"call_{i}"
        if id_column is not None and pd.isna(table[id_column].iloc[i]):
            raise ValueError("occurrence identity cannot be missing")
        score = table[score_column].iloc[i] if score_column is not None else None
        occurrences.append(UnitOccurrence(occurrence_id, str(type_id),
                                          FrameSpan(table[start_column].iloc[i], table[stop_column].iloc[i]), score, score_kind))
    return UnitStream(session, level, model_id, clock, fps, tuple(valid_intervals), tuple(catalog), tuple(occurrences))


def from_bout_windows(windows: pd.DataFrame, bouts: pd.DataFrame, *, session: SessionKey, level: str,
                      model_id: str, clock: str, fps: float, valid_intervals: Iterable[FrameSpan],
                      catalog: Iterable[UnitType], type_column="chain_id", start_column="start", stop_column="end",
                      score_column=None, score_kind=None) -> UnitStream:
    """Map detector windows from bout-row positions to native intervals explicitly.

    Supports exact/seqNMF windows on an unpadded bout view. Padded concatenated
    seqNMF arrays must be mapped to original rows first; no padding or segment
    conversion is inferred here. A window cannot bridge discontinuous bouts or
    declared segment boundaries.
    """
    if not {start_column, stop_column, type_column} <= set(windows):
        raise ValueError("window identity/start/stop columns are required")
    if not {"start_frame", "end_frame_exclusive"} <= set(bouts):
        raise ValueError("bout table requires native-frame intervals")
    native = windows.copy()
    starts, stops = [], []
    for a, b in zip(windows[start_column], windows[stop_column]):
        position = FrameSpan(a, b)
        if position.stop > len(bouts):
            raise ValueError("window exceeds the supplied bout view")
        part = bouts.iloc[position.start:position.stop]
        if (part.start_frame.to_numpy()[1:] != part.end_frame_exclusive.to_numpy()[:-1]).any():
            raise ValueError("window cannot cross missing or unmatched bout intervals")
        if "segment" in part and part.segment.nunique(dropna=False) != 1:
            raise ValueError("window cannot cross a declared segment boundary")
        starts.append(part.start_frame.iloc[0])
        stops.append(part.end_frame_exclusive.iloc[-1])
    # Typed empty columns retain the native-coordinate schema for zero detections.
    native["start_frame"] = pd.Series(starts, index=native.index, dtype="int64") if not starts else starts
    native["end_frame_exclusive"] = pd.Series(stops, index=native.index, dtype="int64") if not stops else stops
    return from_occurrence_table(native, session=session, level=level, model_id=model_id, clock=clock, fps=fps,
                                 valid_intervals=valid_intervals, catalog=catalog, type_column=type_column,
                                 score_column=score_column, score_kind=score_kind)


def occurrence_table(stream: UnitStream) -> pd.DataFrame:
    """Chronological native interval view, retaining original occurrence identity."""
    if not isinstance(stream, UnitStream):
        raise TypeError("a UnitStream is required; adapt the detector output explicitly")
    rows = [{"occurrence_id": event.occurrence_id, "type_id": event.type_id,
             "start_frame": event.interval.start, "end_frame_exclusive": event.interval.stop,
             "score": event.score, "score_kind": event.score_kind}
            for event in stream.occurrences]
    table = pd.DataFrame(rows, columns=["occurrence_id", "type_id", "start_frame", "end_frame_exclusive", "score", "score_kind"])
    return table.sort_values(["start_frame", "end_frame_exclusive", "type_id", "occurrence_id"], kind="stable").reset_index(drop=True)


def primitive_bout_table(stream: UnitStream):
    """Expose a primitive stream to existing bout algorithms with an ID crosswalk.

    Returns (table, code_to_type_id). The catalogue order fixes integer codes;
    codes are not identities across fitted catalogues. No additional RLE is done.
    """
    if stream.level != "primitive":
        raise ValueError("a primitive unit stream is required")
    table = occurrence_table(stream)
    if len(table) and (table.start_frame.to_numpy()[1:] < table.end_frame_exclusive.to_numpy()[:-1]).any():
        raise ValueError("primitive bout algorithms require exclusive occurrences")
    code_to_type_id = tuple(unit.type_id for unit in stream.catalog)
    codes = {type_id: i for i, type_id in enumerate(code_to_type_id)}
    table["syllable_id"] = table.type_id.map(codes).astype(int)
    span_starts = np.array([span.start for span in stream.valid_intervals], dtype=np.int64)
    table["segment"] = np.searchsorted(span_starts, table.start_frame.to_numpy(dtype=np.int64), side="right") - 1
    table["duration_s"] = (table.end_frame_exclusive - table.start_frame) / stream.fps
    # Missing/unassigned time inside a valid source span is also a token barrier.
    if len(table):
        barriers = np.r_[True, (np.diff(table.segment) != 0)
                         | (table.start_frame.to_numpy()[1:] != table.end_frame_exclusive.to_numpy()[:-1])]
        table["segment"] = np.cumsum(barriers) - 1
    return table, code_to_type_id


def occurrence_blocks(stream: UnitStream, *, break_gaps: bool = True, excluded_types=()):
    """Project an exclusive stream into ordered blocks without losing barriers.

    Touching declared validity spans remain distinct. Exclusions are applied to
    inclusive occurrence order, so dropping B from A B A cannot join A to A.
    Sparse event context may explicitly keep unmatched valid time between events.
    """
    from bisect import bisect_right

    if not isinstance(stream, UnitStream):
        raise ValueError("occurrence blocks require a UnitStream")
    ordered = sorted(stream.occurrences, key=lambda o: (o.interval.start, o.interval.stop))
    if any(a.interval.stop > b.interval.start for a, b in zip(ordered, ordered[1:])):
        raise ValueError("occurrence blocks require an explicit exclusive view")
    starts = [v.start for v in stream.valid_intervals]
    excluded = frozenset(excluded_types)
    result, current = [], []
    previous_span = None
    for occ in ordered:
        span = bisect_right(starts, occ.interval.start)-1
        if occ.type_id in excluded:
            if current:
                result.append(tuple(current))
            current = []
            previous_span = None
            continue
        if current and (span != previous_span or (break_gaps and current[-1].interval.stop < occ.interval.start)):
            result.append(tuple(current))
            current = []
        current.append(occ)
        previous_span = span
    if current:
        result.append(tuple(current))
    return tuple(result)
