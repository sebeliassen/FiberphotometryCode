"""Continuous observation and trajectory representations from pose tracking.

Provides extraction of continuous kinematic features (speed, planar heading/yaw,
displacement, and path length) from explicit PoseSeries and TimeAxis records.
Preserves raw features, native coordinates, and per-feature validity masks without
leaking across unobserved frames or missing support.
"""

from __future__ import annotations

import json
import math
from numbers import Real

import numpy as np

from fpcode.contracts.behavior_results import FeatureAssociation
from fpcode.contracts.pose import PoseSeries
from fpcode.contracts.sessions import SessionKey
from fpcode.contracts.signals import SignalMetadata, SignalSeries

IMPLEMENTATION_STATUS = "implemented"


def _circular_difference(rad_next: np.ndarray, rad_prev: np.ndarray) -> np.ndarray:
    """Compute circular difference in [-pi, pi], avoiding wrap jumps."""
    diff = rad_next - rad_prev
    return (diff + np.pi) % (2.0 * np.pi) - np.pi


def robust_position_bounds(positions, *, valid=None, quantiles=(.005, .995)):
    """Axis-aligned empirical position envelope, explicitly a geometry proxy.

    The caller selects the reference recordings/support; this estimator does not
    know physical walls and must not silently normalise each comparison animal.
    """
    positions = np.asarray(positions, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (samples, 2)")
    if q.shape != (2,) or not np.isfinite(q).all() or not 0 <= q[0] < q[1] <= 1:
        raise ValueError("quantiles must be an ordered pair in [0, 1]")
    supported = np.isfinite(positions).all(axis=1)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != supported.shape:
            raise ValueError("position validity must match samples")
        supported &= valid
    if not supported.any():
        raise ValueError("no supported positions for empirical geometry proxy")
    low, high = np.quantile(positions[supported], q, axis=0)
    if np.any(high <= low):
        raise ValueError("position envelope must have positive extent")
    return float(low[0]), float(high[0]), float(low[1]), float(high[1])


def append_landmark_centroid(pose: PoseSeries, *, bodyparts, name, confidence_threshold, minimum_count=2):
    """Append a supported anatomical-reference average; never a camera correction.

    Eligible landmarks retain their physical coordinates. The derived confidence
    is their minimum, and insufficient support remains missing. Callers choose
    which points are scientifically comparable and how many are required.
    """
    parts = tuple(bodyparts)
    if not isinstance(pose, PoseSeries) or not parts or len(set(parts)) != len(parts):
        raise ValueError("unique source bodyparts and PoseSeries are required")
    if not isinstance(name, str) or not name or name in pose.bodyparts or any(part not in pose.bodyparts for part in parts):
        raise ValueError("derived name must be new and source bodyparts must exist")
    if (isinstance(minimum_count, bool) or not isinstance(minimum_count, int)
            or not 1 <= minimum_count <= len(parts)
            or not isinstance(confidence_threshold, Real) or not 0 <= confidence_threshold <= 1):
        raise ValueError("minimum_count and confidence_threshold require valid support bounds")
    indices = [pose.bodyparts.index(part) for part in parts]
    coords, confidence = pose.coordinates[:, indices], pose.confidence[:, indices]
    eligible = pose.valid[:, indices] & (confidence >= confidence_threshold)
    count = eligible.sum(axis=1)
    supported = count >= minimum_count
    centroid = np.sum(np.where(eligible[:, :, None], coords, 0.), axis=1) / np.maximum(count[:, None], 1)
    derived_confidence = np.min(np.where(eligible, confidence, np.inf), axis=1)
    centroid[~supported] = np.nan
    derived_confidence[~supported] = np.nan
    return PoseSeries(pose.time, np.concatenate([pose.coordinates, centroid[:, None]], axis=1),
        np.column_stack([pose.confidence, derived_confidence]), (*pose.bodyparts, name),
        pose.unit, pose.coordinate_frame, np.column_stack([pose.valid, supported]))


def _rectangle_distances(positions, supported, bounds, include_corners=False):
    xmin, xmax, ymin, ymax = bounds
    x, y = positions[:, 0], positions[:, 1]
    inside = supported & (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    dx = np.minimum(np.abs(x - xmin), np.abs(x - xmax))
    dy = np.minimum(np.abs(y - ymin), np.abs(y - ymax))
    distances = np.column_stack([
        np.hypot(dx, dy), np.minimum(dx, dy),
        np.hypot(x - (xmin + xmax) / 2, y - (ymin + ymax) / 2),
    ])
    if include_corners:
        distances = np.column_stack([distances, *(
            np.hypot(x - cx, y - cy) for cx, cy in (
                (xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax),
            ))])
    distances[~inside] = np.nan
    return distances, np.repeat(inside[:, None], distances.shape[1], axis=1)


def extract_trajectory_features(
    pose: PoseSeries,
    *,
    fps: float | None = None,
    anterior_landmark: str = "nose",
    posterior_landmark: str = "tail_base",
    confidence_threshold: float | None = None,
    smooth_window: int | None = None,
    max_gap_s: float | None = None,
    arena_bounds: tuple[float, float, float, float] | None = None,
    arena_landmarks: tuple[str, ...] = (),
    arena_corner_features: bool = False,
) -> SignalSeries:
    """Extract physical speed, heading, and angular velocity from PoseSeries.

    Parameters
    ----------
    pose : PoseSeries
        Pose series containing coordinates, confidence, bodyparts, and time axis.
    fps : float, optional
        Sampling rate in Hz. Required if pose.time.unit == "frame".
    anterior_landmark : str
        Bodypart name for anterior reference (e.g. "nose").
    posterior_landmark : str
        Bodypart name for posterior reference (e.g. "tail_base").
    confidence_threshold : float, optional
        Minimum confidence required for a landmark to be considered valid.
    smooth_window : int, optional
        Optional odd window length for median filtering within contiguous blocks.
    arena_bounds : tuple, optional
        Explicit rectangular arena (xmin, xmax, ymin, ymax), axis-aligned in
        pose.coordinate_frame and pose.unit. Adds distances from the body
        midpoint to the nearest corner, finite edge and rectangle centre.
        Geometry is never inferred from observed mouse positions. Outside-arena
        observations have invalid distance channels; other channels are unchanged.
    arena_landmarks : tuple
        Additional tracked reference points for distance channels, e.g. ("nose",).
        Their confidence/support is independent of midpoint support. Channels are
        prefixed by the landmark name; midpoint channels retain their original names.
    arena_corner_features : bool
        Also retain each image-coordinate corner distance (top-left, top-right,
        bottom-left, bottom-right; assumes image y increases downwards). Camera
        relative corner interpretation is supplied by the owning study.

    Returns
    -------
    SignalSeries
        Multi-channel continuous signal series containing:
        - centroid_x, centroid_y (position)
        - speed (displacement / dt)
        - heading (planar angle in [-pi, pi])
        - angular_velocity (d(heading)/dt)
    """
    if not isinstance(pose, PoseSeries):
        raise ValueError("pose must be a PoseSeries")
    if fps is not None and (isinstance(fps, bool) or not isinstance(fps, Real)
                            or not math.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be finite and positive")
    if max_gap_s is not None and (not math.isfinite(max_gap_s) or max_gap_s <= 0):
        raise ValueError("max_gap_s must be finite and positive")
    bounds = None
    if arena_bounds is not None:
        bounds = np.asarray(arena_bounds, dtype=float)
        if (bounds.shape != (4,) or not np.all(np.isfinite(bounds))
                or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]):
            raise ValueError("arena_bounds must be finite (xmin, xmax, ymin, ymax) with positive extent")

    parts = list(pose.bodyparts)
    arena_landmarks = tuple(arena_landmarks)
    if len(set(arena_landmarks)) != len(arena_landmarks) or any(name not in parts for name in arena_landmarks):
        raise ValueError("arena_landmarks must name unique tracked bodyparts")
    if arena_landmarks and bounds is None:
        raise ValueError("arena_landmarks require explicit arena_bounds")
    if not isinstance(arena_corner_features, bool) or (arena_corner_features and bounds is None):
        raise ValueError("arena_corner_features must be boolean and requires arena_bounds")
    if anterior_landmark not in parts:
        raise ValueError(f"anterior landmark '{anterior_landmark}' not in bodyparts: {parts}")
    if posterior_landmark not in parts:
        raise ValueError(f"posterior landmark '{posterior_landmark}' not in bodyparts: {parts}")
    ant_idx = parts.index(anterior_landmark)
    post_idx = parts.index(posterior_landmark)

    # Resolve delta t (dt) in seconds
    t_values = np.asarray(pose.time.values, dtype=float)
    n_frames = len(t_values)
    if n_frames < 1:
        raise ValueError("pose series cannot be empty")

    if pose.time.unit == "frame":
        if fps is None or not isinstance(fps, Real) or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("positive fps is required when time unit is 'frame'")
        effective_fps = float(fps)
        # Frame diffs
        frame_diffs = np.zeros(n_frames, dtype=float)
        if n_frames > 1:
            frame_diffs[1:] = np.diff(t_values)
        dt = np.where(frame_diffs > 0, frame_diffs / effective_fps, 1.0 / effective_fps)
        is_contiguous = np.ones(n_frames, dtype=bool)
        if n_frames > 1:
            is_contiguous[1:] = (frame_diffs[1:] == 1.0)
        is_contiguous[0] = False  # first sample has no backwards step
    elif pose.time.unit in {"s", "ms", "ns"}:
        scale_to_s = {"s": 1.0, "ms": 1e-3, "ns": 1e-9}[pose.time.unit]
        t_sec = t_values * scale_to_s
        dt = np.zeros(n_frames, dtype=float)
        if n_frames > 1:
            dt[1:] = np.diff(t_sec)
        if fps is not None:
            expected_dt = 1.0 / float(fps)
            is_contiguous = np.zeros(n_frames, dtype=bool)
            if n_frames > 1:
                # Contiguous if within 50% tolerance of expected dt
                is_contiguous[1:] = (np.abs(dt[1:] - expected_dt) <= 0.5 * expected_dt)
        elif max_gap_s is not None:
            is_contiguous = np.zeros(n_frames, dtype=bool)
            if n_frames > 1:
                is_contiguous[1:] = (dt[1:] > 0) & (dt[1:] <= max_gap_s)
        else:
            raise ValueError("physical-time pose requires fps or an explicit max_gap_s support policy")
    else:
        raise ValueError(f"unsupported time unit: {pose.time.unit}")

    # Extract coordinates (T, K, 2)
    coords = np.asarray(pose.coordinates, dtype=float)
    conf = np.asarray(pose.confidence, dtype=float)

    # Per-part validity
    ant_conf_valid = np.ones(n_frames, dtype=bool) if confidence_threshold is None else (conf[:, ant_idx] >= confidence_threshold)
    post_conf_valid = np.ones(n_frames, dtype=bool) if confidence_threshold is None else (conf[:, post_idx] >= confidence_threshold)
    ant_valid = ant_conf_valid & np.all(np.isfinite(coords[:, ant_idx, :2]), axis=1)
    post_valid = post_conf_valid & np.all(np.isfinite(coords[:, post_idx, :2]), axis=1)

    if pose.valid is not None:
        pose_valid = np.asarray(pose.valid, dtype=bool)
        if pose_valid.ndim == 2:
            ant_valid = ant_valid & pose_valid[:, ant_idx]
            post_valid = post_valid & pose_valid[:, post_idx]
        else:
            ant_valid = ant_valid & pose_valid
            post_valid = post_valid & pose_valid

    # Centroid: midpoint between anterior and posterior landmarks
    ant_coords = coords[:, ant_idx, :2]
    post_coords = coords[:, post_idx, :2]
    centroid = 0.5 * (ant_coords + post_coords)
    centroid_valid = ant_valid & post_valid

    # Speed calculation (backward difference)
    speed = np.full(n_frames, np.nan, dtype=float)
    speed_valid = np.zeros(n_frames, dtype=bool)
    if n_frames > 1:
        step_valid = is_contiguous[1:] & centroid_valid[1:] & centroid_valid[:-1] & (dt[1:] > 0)
        disp = np.linalg.norm(centroid[1:] - centroid[:-1], axis=1)
        speed[1:][step_valid] = disp[step_valid] / dt[1:][step_valid]
        speed_valid[1:] = step_valid

    # Heading in radians [-pi, pi]
    heading = np.full(n_frames, np.nan, dtype=float)
    heading_valid = ant_valid & post_valid
    dx = ant_coords[:, 0] - post_coords[:, 0]
    dy = ant_coords[:, 1] - post_coords[:, 1]
    dist = np.sqrt(dx**2 + dy**2)
    heading_valid = heading_valid & (dist > 1e-6)
    heading[heading_valid] = np.arctan2(dy[heading_valid], dx[heading_valid])

    # Angular velocity (yaw rate)
    ang_vel = np.full(n_frames, np.nan, dtype=float)
    ang_vel_valid = np.zeros(n_frames, dtype=bool)
    if n_frames > 1:
        yaw_step_valid = is_contiguous[1:] & heading_valid[1:] & heading_valid[:-1] & (dt[1:] > 0)
        circ_diff = _circular_difference(heading[1:], heading[:-1])
        ang_vel[1:][yaw_step_valid] = circ_diff[yaw_step_valid] / dt[1:][yaw_step_valid]
        ang_vel_valid[1:] = yaw_step_valid

    # Optional smoothing within contiguous valid runs only (no leakage across gaps)
    if smooth_window is not None and smooth_window > 1:
        win = int(smooth_window)
        if win % 2 == 0:
            win += 1
        speed = _smooth_valid_runs(speed, speed_valid, win)
        ang_vel = _smooth_valid_runs(ang_vel, ang_vel_valid, win)

    values = np.column_stack([
        centroid[:, 0],
        centroid[:, 1],
        np.nan_to_num(speed, nan=0.0),
        np.nan_to_num(heading, nan=0.0),
        np.nan_to_num(ang_vel, nan=0.0),
    ])
    valid_mask = np.column_stack([
        centroid_valid,
        centroid_valid,
        speed_valid,
        heading_valid,
        ang_vel_valid,
    ])

    unit = pose.unit
    signals = (
        SignalMetadata(name="centroid_x", unit=unit),
        SignalMetadata(name="centroid_y", unit=unit),
        SignalMetadata(name="speed", unit=f"{unit}/s"),
        SignalMetadata(name="heading", unit="rad"),
        SignalMetadata(name="angular_velocity", unit="rad/s"),
    )

    if bounds is not None:
        distance_names = ("distance_nearest_corner", "distance_nearest_edge", "distance_center")
        if arena_corner_features:
            distance_names += tuple(f"distance_{corner}_corner" for corner in (
                "top_left", "top_right", "bottom_left", "bottom_right",
            ))
        distances, distance_valid = _rectangle_distances(centroid, centroid_valid, bounds, arena_corner_features)
        values = np.column_stack([values, distances])
        valid_mask = np.column_stack([valid_mask, distance_valid])
        signals += tuple(SignalMetadata(name=name, unit=unit) for name in distance_names)
        for landmark in arena_landmarks:
            idx = parts.index(landmark)
            supported = np.isfinite(coords[:, idx, :2]).all(axis=1)
            if confidence_threshold is not None:
                supported &= conf[:, idx] >= confidence_threshold
            supported &= pose.valid[:, idx]
            distances, distance_valid = _rectangle_distances(coords[:, idx, :2], supported, bounds, arena_corner_features)
            values = np.column_stack([values, distances])
            valid_mask = np.column_stack([valid_mask, distance_valid])
            signals += tuple(SignalMetadata(name=f"{landmark}_{name}", unit=unit) for name in distance_names)

    return SignalSeries(
        time=pose.time,
        values=values,
        signals=signals,
        valid=valid_mask,
    )


def _smooth_valid_runs(data: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    """Apply median filter strictly within contiguous True runs of mask."""
    from scipy.ndimage import median_filter

    out = data.copy()
    if not np.any(mask):
        return out
    runs = np.flatnonzero(np.diff(np.pad(mask.astype(np.int8), (1, 1), mode="constant")))
    starts = runs[::2]
    stops = runs[1::2]
    for s, e in zip(starts, stops):
        if e - s >= window:
            out[s:e] = median_filter(data[s:e], size=window, mode="nearest")
    return out


def native_feature_coordinates(features: SignalSeries, native_frames=None) -> np.ndarray:
    """Resolve declared native samples; never guess array rows or a clock origin."""
    if native_frames is None:
        if features.time.unit != "frame":
            raise ValueError("physical-time features require an explicit native frame mapping")
        native_frames = features.time.values
    frames = np.asarray(native_frames)
    if (frames.shape != (len(features.time),) or not np.issubdtype(frames.dtype, np.number)
            or np.any(~np.isfinite(frames)) or np.any(frames < 0)
            or np.any(frames != np.floor(frames))
            or np.any(frames.astype(np.longdouble) > np.iinfo(np.int64).max)):
        raise ValueError("native samples must be exactly representable nonnegative int64 frames")
    frames = frames.astype(np.int64, copy=False)
    if np.any(frames[1:] <= frames[:-1]):
        raise ValueError("native frame mapping must be strictly increasing")
    if features.time.unit == "frame" and not np.array_equal(frames, features.time.values):
        raise ValueError("native mapping disagrees with feature frame coordinates")
    return frames


def extract_associated_trajectory_features(pose: PoseSeries, *, session: SessionKey,
                                         native_frames=None, **options) -> FeatureAssociation:
    """Extract features with recording, mapping and explicit extraction provenance."""
    series = extract_trajectory_features(pose, **options)
    return FeatureAssociation(
        series=series, session=session, coordinate_frame=pose.coordinate_frame,
        native_frames=native_frames,
        provenance=json.dumps({"operation": "trajectory_features",
                               "options": {key: np.asarray(value).tolist() if key == "arena_bounds"
                                           and value is not None else value
                                           for key, value in options.items()},
                               "landmarks": list(pose.bodyparts), "position_unit": pose.unit,
                               "clock": pose.time.clock, "time_unit": pose.time.unit}, sort_keys=True),
    )


def compute_interval_trajectory_metrics(
    features: SignalSeries,
    start_frame: int,
    stop_frame: int,
    *,
    native_frames=None,
) -> dict[str, float]:
    """Compute trajectory summary metrics (path length, net displacement, speed) for an interval.

    Parameters
    ----------
    features : SignalSeries
        Signal series from extract_trajectory_features.
    start_frame : int
        Native frame coordinate (inclusive).
    stop_frame : int
        Native frame coordinate (exclusive).

    Returns
    -------
    dict
        Path length, net displacement, mean speed, max speed, supported samples.
    """
    if start_frame < 0 or stop_frame <= start_frame:
        raise ValueError("invalid interval bounds for features: start_frame must be >= 0 and stop_frame > start_frame")

    time_vals = native_feature_coordinates(features, native_frames)
    lo, hi = np.searchsorted(time_vals, [start_frame, stop_frame])
    span_time = time_vals[lo:hi]
    s_vals = features.values[lo:hi]
    s_valid = features.valid[lo:hi]
    if s_vals.ndim == 1:
        s_vals, s_valid = s_vals[:, None], s_valid[:, None]

    names = [s.name for s in features.signals]
    speed_col = names.index("speed") if "speed" in names else None
    cx_col = names.index("centroid_x") if "centroid_x" in names else None
    cy_col = names.index("centroid_y") if "centroid_y" in names else None

    if speed_col is not None and len(s_vals) > 0:
        span_valid_speed = s_valid[:, speed_col]
        span_speed = s_vals[:, speed_col]
        n_supported = int(np.sum(span_valid_speed))
        mean_speed = float(np.mean(span_speed[span_valid_speed])) if n_supported > 0 else float("nan")
        max_speed = float(np.max(span_speed[span_valid_speed])) if n_supported > 0 else float("nan")
    else:
        n_supported = 0
        mean_speed = float("nan")
        max_speed = float("nan")

    path_len = 0.0
    net_disp = float("nan")
    if cx_col is not None and cy_col is not None and len(s_vals) > 0:
        span_cx = s_vals[:, cx_col]
        span_cy = s_vals[:, cy_col]
        span_valid_pos = s_valid[:, cx_col] & s_valid[:, cy_col]

        # Path length: sum across contiguous native steps (dt == 1) only
        if len(span_time) > 1:
            dt = np.diff(span_time)
            step_valid = span_valid_pos[1:] & span_valid_pos[:-1] & (dt == 1)
            dx = np.diff(span_cx)
            dy = np.diff(span_cy)
            step_disp = np.sqrt(dx**2 + dy**2)
            path_len = float(np.sum(step_disp[step_valid]))

        valid_indices = np.flatnonzero(span_valid_pos)
        if len(valid_indices) >= 2:
            first_idx, last_idx = valid_indices[0], valid_indices[-1]
            net_disp = float(np.sqrt((span_cx[last_idx] - span_cx[first_idx])**2 + (span_cy[last_idx] - span_cy[first_idx])**2))
        elif len(valid_indices) == 1:
            net_disp = 0.0

    return {
        "supported_samples": float(n_supported),
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "path_length": path_len,
        "net_displacement": net_disp,
    }


__all__ = [
    "IMPLEMENTATION_STATUS",
    "extract_trajectory_features",
    "extract_associated_trajectory_features",
    "native_feature_coordinates",
    "compute_interval_trajectory_metrics",
]
