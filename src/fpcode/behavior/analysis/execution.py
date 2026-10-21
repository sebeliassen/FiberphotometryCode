"""Shared native-time execution, onset/phase profiles and progression measurements.

Each feature sample represents a supported native frame, never an invented row
coordinate or interpolated gap. Profiles average observed frame cells; physical
pacing remains separate from phase-normalized shape.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import pandas as pd

from fpcode.behavior.observations.representations import native_feature_coordinates
from fpcode.contracts.behavior_results import BehaviorResult, FeatureAssociation
from fpcode.contracts.behavior_units import UnitStream
from fpcode.contracts.signals import SignalSeries

IMPLEMENTATION_STATUS = "implemented"


def _unwrap(stream_or_result, features):
    result = stream_or_result if isinstance(stream_or_result, BehaviorResult) else None
    stream = result.stream if result is not None else stream_or_result
    if not isinstance(stream, UnitStream):
        raise ValueError("stream_or_result must be a UnitStream or BehaviorResult")
    if features is None and result is not None:
        features = result.features
    if features is None:
        return stream, result, None
    association = features if isinstance(features, FeatureAssociation) else None
    series = association.series if association is not None else features
    if not isinstance(series, SignalSeries):
        raise ValueError("features must be a SignalSeries or FeatureAssociation")
    if association is not None and association.session != stream.session:
        raise ValueError("feature recording does not match stream session")
    if series.time.clock != stream.clock:
        raise ValueError("feature clock does not match stream clock")
    frames = native_feature_coordinates(series, None if association is None else association.native_frames)
    if series.time.unit != "frame" and len(frames) > 1:
        scale = {"s": 1., "ms": 1e-3, "ns": 1e-9}[series.time.unit]
        seconds = np.diff(series.time.values.astype(np.longdouble)) * scale
        if not np.allclose(seconds, np.diff(frames) / stream.fps, rtol=1e-6, atol=1e-9):
            raise ValueError("physical feature times disagree with native clock/fps mapping")
    weights = np.full(len(frames), 1. / stream.fps)
    if association is not None and association.sample_exposure_s is not None:
        weights = association.sample_exposure_s
        if np.any(weights > 1. / stream.fps + 1e-12):
            raise ValueError("frame exposure cannot exceed one native frame; use an explicit interval adapter")
    values, valid = series.values, series.valid
    if values.ndim == 1:
        values, valid = values[:, None], valid[:, None]
    return stream, result, (series, frames, values, valid, weights)


def _column(series, name):
    names = [s.name for s in series.signals]
    if name not in names:
        raise ValueError(f"required feature {name} is unavailable")
    return names.index(name)


def _summary(values, valid, weights):
    supported = valid & (weights > 0)
    count = int(supported.sum())
    exposure = float(weights[supported].sum())
    if not count:
        return dict(supported_samples=0, exposure_s=0., mean_feature=np.nan,
                    sd_feature=np.nan, peak_feature=np.nan, feature_integral=0.)
    x, w = values[supported], weights[supported]
    mean = float(np.sum(x * w) / exposure)
    return dict(supported_samples=count, exposure_s=exposure, mean_feature=mean,
                sd_feature=float(np.sqrt(np.sum(w * (x - mean) ** 2) / exposure)),
                peak_feature=float(x.max()), feature_integral=float(np.sum(x * w)))


def measure_occurrence_execution(stream_or_result: UnitStream | BehaviorResult,
                                 features: SignalSeries | FeatureAssociation | None = None,
                                 *, feature_name: str = "speed") -> pd.DataFrame:
    """Measure native half-open occurrences with explicit support and evidence.

    Bare frame-coordinate SignalSeries inputs assert recording association by
    their caller. Use FeatureAssociation for validation of recording identity.
    Physical-time series require an explicit native mapping in that association.
    """
    stream, result, view = _unwrap(stream_or_result, features)
    comp = {} if result is None or result.completion is None else {c.occurrence_id: c for c in result.completion}
    phase_counts = {}
    if result is not None and result.phases is not None:
        for phase in result.phases:
            phase_counts[phase.parent_occurrence_id] = phase_counts.get(phase.parent_occurrence_id, 0) + 1
    if view is not None:
        series, frames, values, valid, weights = view
        col = _column(series, feature_name)
        names = [s.name for s in series.signals]
        xy = [names.index(n) for n in ("centroid_x", "centroid_y")] if {"centroid_x", "centroid_y"} <= set(names) else None
        if xy is not None and series.signals[xy[0]].unit != series.signals[xy[1]].unit:
            raise ValueError("position channels require matching physical units")
    rows = []
    for occ in stream.occurrences:
        start, stop = occ.interval.start, occ.interval.stop
        row = dict(occurrence_id=occ.occurrence_id, type_id=occ.type_id,
                   start_frame=start, stop_frame=stop, duration_frames=stop-start,
                   duration_s=(stop-start)/stream.fps, score=occ.score, score_kind=occ.score_kind,
                   completion_status="unknown", observed_entry=None, observed_exit=None,
                   final_phase_reached=None, left_censored=None, right_censored=None,
                   completion_evidence_kind=None, phase_count=phase_counts.get(occ.occurrence_id, 0))
        if view is not None:
            lo, hi = np.searchsorted(frames, [start, stop])
            f, x, v, w = frames[lo:hi], values[lo:hi, col], valid[lo:hi, col], weights[lo:hi]
            row.update(_summary(x, v, w))
            row.update(feature_name=feature_name, feature_unit=series.signals[col].unit)
            supported = np.flatnonzero(v & (w > 0))
            peak = f[supported[np.argmax(x[supported])]] if len(supported) else np.nan
            row.update(peak_latency_frames=float(peak-start), peak_latency_s=float((peak-start)/stream.fps),
                       path_length=np.nan, net_displacement=np.nan, path_supported_steps=0)
            if xy is not None:
                p = values[lo:hi][:, xy]
                pv = np.all(valid[lo:hi][:, xy], axis=1)
                locs = np.flatnonzero(pv)
                steps = pv[1:] & pv[:-1] & (np.diff(f) == 1)
                row["path_supported_steps"] = int(steps.sum())
                if len(locs):
                    row["path_length"] = float(np.linalg.norm(np.diff(p, axis=0)[steps], axis=1).sum())
                    row["net_displacement"] = float(np.linalg.norm(p[locs[-1]]-p[locs[0]]))
                row["position_supported_samples"] = len(locs)
                row["position_unit"] = series.signals[xy[0]].unit
        evidence = comp.get(occ.occurrence_id)
        if evidence is not None:
            row.update(completion_status=evidence.status, observed_entry=evidence.observed_entry,
                       observed_exit=evidence.observed_exit, final_phase_reached=evidence.final_phase_reached,
                       left_censored=evidence.left_censored, right_censored=evidence.right_censored,
                       completion_evidence_kind=evidence.evidence_kind)
        rows.append(row)
    columns = ["occurrence_id", "type_id", "start_frame", "stop_frame", "duration_frames", "duration_s",
               "score", "score_kind", "completion_status", "observed_entry", "observed_exit",
               "final_phase_reached", "left_censored", "right_censored", "completion_evidence_kind", "phase_count"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


def aggregate_execution_metrics(occurrence_metrics: pd.DataFrame) -> pd.DataFrame:
    """Pool feature exposure; keep unevaluable completion separate from failure."""
    rows = []
    for type_id, group in occurrence_metrics.groupby("type_id"):
        row = dict(type_id=type_id, count=len(group), total_duration_s=float(group.duration_s.sum()),
                   mean_duration_s=float(group.duration_s.mean()),
                   sd_duration_s=float(group.duration_s.std()) if len(group) > 1 else 0.)
        if "exposure_s" in group:
            exposure = float(group.exposure_s.sum())
            row.update(total_supported_samples=float(group.supported_samples.sum()), total_exposure_s=exposure,
                       pooled_feature_mean=float(group.feature_integral.sum()/exposure) if exposure else np.nan)
        if "completion_status" in group:
            status = group.completion_status
            evaluable = status.isin(["completed", "deviated"])
            if "observed_exit" in group:
                evaluable &= group.observed_exit.eq(True)
            if "left_censored" in group:
                evaluable &= ~group.left_censored.eq(True)
            n = int(evaluable.sum())
            completed = int(((status == "completed") & evaluable).sum())
            row.update(completed_count=completed, deviated_count=int(((status == "deviated") & evaluable).sum()),
                       censored_count=int((status == "censored").sum()), unknown_count=int((status == "unknown").sum()),
                       evaluable_count=n, completion_rate=float(completed/n) if n else np.nan)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["type_id", "count", "total_duration_s", "mean_duration_s"])


def measure_phase_pacing(result: BehaviorResult) -> pd.DataFrame:
    """Physical phase duration/fraction from validated phases; no shape rescaling."""
    if not isinstance(result, BehaviorResult) or result.phases is None:
        raise ValueError("measure_phase_pacing requires a BehaviorResult with validated phases")
    parents = {o.occurrence_id: o for o in result.occurrences}
    rows = []
    for p in sorted(result.phases, key=lambda p: (p.parent_occurrence_id, p.order)):
        parent = parents[p.parent_occurrence_id]
        duration = p.interval.stop-p.interval.start
        rows.append(dict(parent_occurrence_id=p.parent_occurrence_id, type_id=parent.type_id,
                         phase_id=p.phase_id, order=p.order, start_frame=p.interval.start,
                         stop_frame=p.interval.stop, duration_frames=duration,
                         duration_s=duration/result.fps,
                         fraction_of_occurrence=duration/(parent.interval.stop-parent.interval.start),
                         evidence_kind=p.evidence_kind))
    return pd.DataFrame(rows, columns=["parent_occurrence_id", "type_id", "phase_id", "order", "start_frame",
                                      "stop_frame", "duration_frames", "duration_s", "fraction_of_occurrence", "evidence_kind"])


def _profile_bin(view, col, low, high):
    series, frames, values, valid, weights = view
    lo = np.searchsorted(frames, low-1., side="right")
    hi = np.searchsorted(frames, high, side="left")
    f = frames[lo:hi]
    overlap = np.maximum(0., np.minimum(f+1., high)-np.maximum(f, low))
    return _summary(values[lo:hi, col], valid[lo:hi, col], weights[lo:hi]*overlap)


def _profile_budget(stream, view, max_rows):
    if isinstance(max_rows, bool) or not isinstance(max_rows, Integral) or max_rows < 1:
        raise ValueError("profile row budget must be a positive integer")
    if any(o.interval.stop > 2**53 for o in stream.occurrences) or (len(view[1]) and view[1][-1] > 2**53):
        raise ValueError("profile coordinates exceed exact floating integer representation")


def measure_onset_profiles(stream_or_result, features=None, *, feature_name="speed",
                           before_s=0., after_s=1., bin_s=.1, max_rows=250000) -> pd.DataFrame:
    """Average supported frame cells in physical bins around each native onset.

    Profiles can include neighbouring units, but never cross the onset's declared
    validity block. No interpolation or extrapolation fills unsupported frames.
    """
    stream, _, view = _unwrap(stream_or_result, features)
    if view is None:
        raise ValueError("onset profiles require associated features")
    _profile_budget(stream, view, max_rows)
    if isinstance(max_rows, bool) or not isinstance(max_rows, Integral) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    if not all(math.isfinite(x) for x in (before_s, after_s, bin_s)) or before_s < 0 or after_s <= 0 or bin_s <= 0:
        raise ValueError("profile durations/bin width must be finite and positive; before_s can be zero")
    n = math.ceil((before_s+after_s)/bin_s)
    if n > 4096 or n*len(stream.occurrences) > max_rows:
        raise ValueError("onset profile output exceeds the configured row/bin budget")
    col = _column(view[0], feature_name)
    rows = []
    for occ in stream.occurrences:
        span = next(v for v in stream.valid_intervals if v.start <= occ.interval.start and occ.interval.stop <= v.stop)
        for i in range(n):
            a, b = -before_s+i*bin_s, min(-before_s+(i+1)*bin_s, after_s)
            low, high = max(span.start, occ.interval.start+a*stream.fps), min(span.stop, occ.interval.start+b*stream.fps)
            summary = _profile_bin(view, col, low, high) if high > low else _summary(np.empty(0), np.empty(0, bool), np.empty(0))
            rows.append(dict(occurrence_id=occ.occurrence_id, type_id=occ.type_id, bin_index=i,
                             relative_start_s=a, relative_stop_s=b, feature_name=feature_name,
                             feature_unit=view[0].signals[col].unit, **summary))
    columns = ["occurrence_id", "type_id", "bin_index", "relative_start_s", "relative_stop_s",
               "feature_name", "feature_unit", "supported_samples", "exposure_s", "mean_feature",
               "sd_feature", "peak_feature", "feature_integral"]
    return pd.DataFrame(rows, columns=columns)


def measure_phase_profiles(result: BehaviorResult, features=None, *, feature_name="speed",
                           n_bins=16, max_rows=250000) -> pd.DataFrame:
    """Bin each observed phase by normalized position, retaining physical pacing."""
    if not isinstance(result, BehaviorResult) or result.phases is None:
        raise ValueError("phase profiles require a BehaviorResult with validated phases")
    stream, _, view = _unwrap(result, features)
    if view is None:
        raise ValueError("phase profiles require associated features")
    _profile_budget(stream, view, max_rows)
    if isinstance(max_rows, bool) or not isinstance(max_rows, Integral) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or n_bins < 1 or n_bins > 4096:
        raise ValueError("n_bins must be an integer between 1 and 4096")
    if len(result.phases)*n_bins > max_rows:
        raise ValueError("phase profile output exceeds the configured row budget")
    parents = {o.occurrence_id: o for o in result.occurrences}
    col = _column(view[0], feature_name)
    rows = []
    for p in result.phases:
        duration = p.interval.stop-p.interval.start
        for i in range(n_bins):
            a, b = p.interval.start+duration*i/n_bins, p.interval.start+duration*(i+1)/n_bins
            rows.append(dict(parent_occurrence_id=p.parent_occurrence_id, type_id=parents[p.parent_occurrence_id].type_id,
                             phase_id=p.phase_id, order=p.order, bin_index=i,
                             normalized_start=i/n_bins, normalized_stop=(i+1)/n_bins,
                             phase_duration_s=duration/stream.fps, feature_name=feature_name,
                             feature_unit=view[0].signals[col].unit, **_profile_bin(view, col, a, b)))
    columns = ["parent_occurrence_id", "type_id", "phase_id", "order", "bin_index", "normalized_start",
               "normalized_stop", "phase_duration_s", "feature_name", "feature_unit", "supported_samples",
               "exposure_s", "mean_feature", "sd_feature", "peak_feature", "feature_integral"]
    return pd.DataFrame(rows, columns=columns)


__all__ = ["IMPLEMENTATION_STATUS", "measure_occurrence_execution", "aggregate_execution_metrics",
           "measure_phase_pacing", "measure_onset_profiles", "measure_phase_profiles"]
