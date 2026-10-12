"""Explicit confidence thresholds applied to immutable labelled pose data."""

from numbers import Real

import numpy as np

from fpcode.contracts import PoseSeries


def confidence_streak_candidates(scores, frames, *, fps, threshold, minimum_seconds,
                                  bridge_seconds=0.):
    """Nominate long unsupported-confidence intervals on explicit native frames.

    Scores below threshold or nonfinite are unsupported. Optional bridging joins
    short supported interruptions, never missing native frames. Returned half-open
    intervals are review candidates, not evidence of handling or final exclusions.
    Aggregation across bodyparts/cameras and thresholds remain caller policy.
    """
    frames, scores = np.asarray(frames), np.asarray(scores, dtype=float)
    if (frames.ndim != 1 or frames.dtype.kind not in 'iu' or scores.shape != frames.shape
            or np.any(frames < 0) or np.any(frames[1:] <= frames[:-1])
            or (len(frames) and int(frames[-1]) >= np.iinfo(np.int64).max)):
        raise ValueError('scores require strictly increasing nonnegative native int64 frame coordinates')
    for value, name in ((fps, 'fps'), (threshold, 'threshold'), (minimum_seconds, 'minimum_seconds'),
                        (bridge_seconds, 'bridge_seconds')):
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f'{name} must be finite numeric')
    if fps <= 0 or not 0 <= threshold <= 1 or minimum_seconds <= 0 or bridge_seconds < 0:
        raise ValueError('confidence threshold, duration and FPS bounds are invalid')
    if not len(frames):
        return []
    low = ~np.isfinite(scores) | (scores < threshold)
    blocks = np.cumsum(np.r_[True, frames[1:] != frames[:-1] + 1])
    boundaries = np.flatnonzero(np.r_[True, (low[1:] != low[:-1]) | (blocks[1:] != blocks[:-1]), True])
    runs = [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:]) if low[a]]
    joined = []
    maximum_bridge = int(np.floor(bridge_seconds * fps))
    for start, stop in runs:
        if joined and blocks[start] == blocks[joined[-1][0]] and start - joined[-1][1] <= maximum_bridge:
            joined[-1] = (joined[-1][0], stop)
        else:
            joined.append((start, stop))
    minimum_frames = int(np.ceil(minimum_seconds * fps))
    cumulative = np.r_[0, np.cumsum(low, dtype=np.int64)]
    result = []
    for start, stop in joined:
        if stop - start < minimum_frames:
            continue
        values = scores[start:stop]
        finite = np.isfinite(values)
        result.append({'start_frame': int(frames[start]), 'end_frame_exclusive': int(frames[stop - 1]) + 1,
            'duration_s': (stop - start) / fps, 'low_frames': int(cumulative[stop] - cumulative[start]),
            'span_frames': stop - start, 'low_fraction': float((cumulative[stop] - cumulative[start]) / (stop - start)),
            'mean_score': float(values[finite].mean()) if finite.any() else np.nan})
    return result


def supported_track_mask(coordinates, confidence, *, threshold):
    """Raw 2D support from finite x/y and finite thresholded confidence.

    Threshold selection belongs to the caller's policy. This array interface
    retains historical visualization support rules and does not fill gaps.
    """
    coordinates = np.asarray(coordinates)
    confidence = np.asarray(confidence)
    finite_xy = np.isfinite(coordinates[..., 0]) & np.isfinite(coordinates[..., 1])
    with np.errstate(invalid="ignore"):
        confident = np.isfinite(confidence) & (confidence >= threshold)
    return finite_xy & confident


def confidence_mask(pose: PoseSeries, *, threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("confidence threshold must be between zero and one")
    return pose.valid & supported_track_mask(pose.coordinates, pose.confidence, threshold=threshold)


def gate_pose(pose: PoseSeries, *, threshold):
    mask = confidence_mask(pose, threshold=threshold)
    return PoseSeries(pose.time, pose.coordinates, pose.confidence, pose.bodyparts,
                      pose.unit, pose.coordinate_frame, mask)
