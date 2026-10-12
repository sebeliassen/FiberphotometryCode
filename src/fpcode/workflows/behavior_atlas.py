"""Compose explicit frame caches into reusable behavioural measurement tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.behavior.analysis.phenotyping import analyze_frames
from fpcode.behavior.primitives.resolutions import remap_states


def mask_frame_cache_intervals(arrays, intervals, *,
                              observation_fields=("centroid", "heading", "latent"),
                              label_fields=("labels",)):
    """Copy a native-frame cache and invalidate explicit half-open intervals.

    Keep native rows/clocks intact. Labels become -1 and declared observations
    become NaN. Callers declare additional observation/label channels explicitly;
    unlisted metadata is copied unchanged. No presence or handling is inferred.
    """
    frames = np.asarray(arrays["frames"])
    spans = np.asarray(intervals)
    if (frames.ndim != 1 or frames.dtype.kind not in 'iu'
            or np.any(frames < 0) or np.any(frames[1:] <= frames[:-1])
            or (len(frames) and frames[-1] > np.iinfo(np.int64).max)):
        raise ValueError("frames must be increasing nonnegative native integers")
    if spans.size == 0:
        spans = np.empty((0, 2), dtype=np.int64)
    if (spans.ndim != 2 or spans.shape[1] != 2 or spans.dtype.kind not in 'iu'
            or np.any(spans < 0) or np.any(spans[:, 1] <= spans[:, 0])
            or np.any(spans > np.iinfo(np.int64).max)):
        raise ValueError("intervals must contain nonnegative integer half-open pairs")
    if 'labels' not in label_fields or set(label_fields) & set(observation_fields):
        raise ValueError("declare labels once and separate label/observation fields")
    result = {key: np.array(value, copy=True) for key, value in arrays.items()}
    excluded = np.zeros(len(frames), dtype=bool)
    for start, end in spans:
        excluded[np.searchsorted(frames, start):np.searchsorted(frames, end)] = True
    for key in label_fields:
        values = result[key]
        if values.shape != frames.shape or values.dtype.kind not in 'if':
            raise ValueError("label fields must be signed numeric vectors aligned with frames")
        values[excluded] = -1
    for key in observation_fields:
        values = result[key]
        if values.ndim < 1 or len(values) != len(frames) or values.dtype.kind not in 'iuf':
            raise ValueError("observation fields must be numeric and aligned with frames")
        if values.dtype.kind != 'f':
            values = values.astype(float)
            result[key] = values
        values[excluded] = np.nan
    previous = np.asarray(arrays.get('analysis_excluded', np.zeros(len(frames), dtype=bool)))
    if previous.shape != frames.shape or previous.dtype.kind != 'b':
        raise ValueError("analysis_excluded must be a boolean vector aligned with frames")
    result['analysis_excluded'] = previous | excluded
    return result


def extract_measurements(sessions, cache_dir, output_dir, *, num_states, pause_speed_px_s,
                         filter_frames=5, seed=101, label_map=None):
    """Measure each selected session; the caller owns cohort and rate policy."""
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bout_dir = output_dir / "bouts_by_session"
    bout_dir.mkdir(exist_ok=True)
    collected = {k: [] for k in ("syllables", "transitions", "phase_profiles", "onset_profiles", "time_bins")}
    summaries = []
    for index, meta in sessions.iterrows():
        with np.load(cache_dir / (meta.session_id + ".npz"), allow_pickle=False) as source:
            arrays = {k: source[k] for k in source}
        if label_map is not None:
            arrays["labels"] = remap_states(arrays["labels"], label_map)
        measured = analyze_frames(arrays, num_states=num_states, pause_speed_px_s=pause_speed_px_s,
                                  filter_frames=filter_frames, seed=seed)
        identity = meta.to_dict()
        for kind in collected:
            table = measured[kind]
            for key in ("session_id", "animal_id", "group", "condition", "week", "chamber"):
                table[key] = identity[key]
            collected[kind].append(table)
        measured["bouts"].to_csv(bout_dir / (meta.session_id + ".csv"), index=False)
        summary = {**identity, **measured["summary"]}
        # Equal-width elapsed bins are a descriptive within-recording change,
        # not a diagnosis of fatigue or habituation.
        bins = measured["time_bins"]
        for feature in ("mean_speed_px_s", "pause_fraction", "entropy_bits", "bouts_per_min"):
            summary[feature + "_elapsed_slope"] = np.polyfit(bins.elapsed_fraction, bins[feature], 1)[0]
        summaries.append(summary)
        print(f"Measurements {index + 1}/{len(sessions)}: {meta.session_id}", flush=True)
    pd.DataFrame(summaries).to_csv(output_dir / "session_metrics.csv", index=False)
    for kind, tables in collected.items():
        pd.concat(tables, ignore_index=True).to_csv(output_dir / (kind + ".csv"), index=False)
