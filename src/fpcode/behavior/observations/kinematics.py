"""
fpcode.behavior.observations.kinematics
==========================
Kinematic computation, heading and centroid smoothing,
activity window subsetting, and presence masking.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import median_filter

from fpcode.statistics.smoothing import circular_median, coordinate_median


def movement_changepoint_score(
    velocity, fs, *, derivative_scale="per_sample", missing="propagate", smoothing_seconds=1.67,
):
    """Cube the forward difference of a globally z-scored, boxcar-smoothed velocity.

    The Datta snippets historically differentiate per sample and propagate NaNs;
    the unified panel differentiates per second and fills z-score NaNs with zero.
    These policies affect peak thresholds and are intentionally explicit. Rows
    are assumed to have fixed spacing 1/fs; this function never joins recordings
    or resamples gaps. Split/resample irregular timelines before calling it.
    """
    from scipy.ndimage import uniform_filter1d

    velocity = np.asarray(velocity, dtype=float)
    if velocity.ndim != 1:
        raise ValueError("velocity must be one-dimensional")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive")
    if not np.isfinite(smoothing_seconds) or smoothing_seconds < 0:
        raise ValueError("smoothing_seconds must be finite and nonnegative")
    if derivative_scale not in {"per_sample", "per_second"} or missing not in {"propagate", "zero"}:
        raise ValueError("unknown movement initiation policy")
    if not len(velocity) or np.all(np.isnan(velocity)):
        return np.zeros_like(velocity)
    std = np.nanstd(velocity)
    if std == 0 or np.isnan(std):
        return np.zeros_like(velocity)
    standardized = (velocity - np.nanmean(velocity)) / std
    if missing == "zero":
        standardized = np.nan_to_num(standardized, nan=0.0)
    window = max(1, int(round(smoothing_seconds * fs)))
    smoothed = uniform_filter1d(standardized, size=window, mode="nearest")
    delta = np.diff(smoothed, prepend=smoothed[0])
    if derivative_scale == "per_second":
        delta *= fs
    return delta ** 3


def detect_movement_initiations(velocity, fs, *, derivative_scale="per_sample", missing="propagate"):
    """Return row positions with changepoint height, width and prominence >= 1."""
    from scipy.signal import find_peaks

    score = movement_changepoint_score(velocity, fs, derivative_scale=derivative_scale, missing=missing)
    peaks, _ = find_peaks(score, height=1, width=1, prominence=1)
    return peaks.astype(int)


def legacy_nose_tail_kinematics(nose, tail, *, fps=60.0, velocity_ceiling=300.0, smooth_window=7):
    """Preserve tandem export's forward-difference and repeated-final-step policy.

    Coordinates are raw camera pixels. The midpoint is the nose/tail centroid;
    angles use nose minus tail, with unwrap before angular differentiation.
    Velocity is clipped before a nearest-padded median filter. Confidence gating,
    camera correspondence and original-frame selection belong to the caller.
    """
    nose, tail = np.asarray(nose, dtype=float), np.asarray(tail, dtype=float)
    if nose.ndim != 2 or nose.shape[1] != 2 or tail.shape != nose.shape:
        raise ValueError("nose and tail must have matching (frames, 2) coordinates")
    if len(nose) < 2:
        raise ValueError("forward-difference kinematics require at least two frames")
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(velocity_ceiling) or velocity_ceiling < 0:
        raise ValueError("fps must be positive and velocity_ceiling must be nonnegative")
    if isinstance(smooth_window, bool) or int(smooth_window) != smooth_window or smooth_window < 1:
        raise ValueError("smooth_window must be a positive integer")
    centroid = 0.5 * (nose + tail)
    heading = np.arctan2(nose[:, 1] - tail[:, 1], nose[:, 0] - tail[:, 0])
    dx, dy = np.diff(centroid[:, 0]), np.diff(centroid[:, 1])
    velocity = np.pad(np.sqrt(dx**2 + dy**2) * fps, (0, 1), mode="edge")
    angular = np.pad(np.diff(np.unwrap(heading)) * fps, (0, 1), mode="edge")
    return {"centroid": centroid, "heading": heading, "angular_velocity": angular,
            "velocity_raw": velocity,
            "velocity": median_filter(np.clip(velocity, 0, velocity_ceiling),
                                      size=int(smooth_window), mode="nearest")}


def smooth_heading_circular(theta: np.ndarray, window: int = 11, *, policy: str = "behavior") -> np.ndarray:
    """
    Applies median filtering to heading angles by smoothing complex phasors
    e^(i*theta), avoiding phase wrap discontinuities around +/- pi.
    """
    if policy not in {"behavior", "legacy_fusion"}:
        raise ValueError("unknown smoothing policy")
    if policy == "behavior" and len(theta) <= 2:
        return np.asarray(theta, dtype=float)
    if policy == "behavior" and window % 2 == 0:
        window += 1
    return circular_median(theta, window)


def smooth_centroid_median(c: np.ndarray, window: int = 11, *, policy: str = "behavior") -> np.ndarray:
    """
    Applies axis-wise median filtering to 2D centroid coordinates.
    """
    if policy not in {"behavior", "legacy_fusion"}:
        raise ValueError("unknown smoothing policy")
    if policy == "behavior" and len(c) <= 2:
        return np.asarray(c, dtype=float)
    if policy == "behavior" and window % 2 == 0:
        window += 1
    return coordinate_median(c, window)


def clean_short_runs(mask: np.ndarray, min_true: int = 1, min_false: int = 1, *, coerce_bool: bool = True) -> np.ndarray:
    """Remove ON runs shorter than min_true, and OFF runs shorter than min_false."""
    m = np.asarray(mask, dtype=bool).copy() if coerce_bool else mask.copy()
    if m.size == 0:
        return m

    x = m.astype(np.int8)
    boundaries = np.flatnonzero(np.r_[True, x[1:] != x[:-1], True])
    starts = boundaries[:-1]
    ends = boundaries[1:]
    vals = m[starts]

    if min_true > 1:
        for s, e, v in zip(starts, ends, vals):
            if v and (e - s) < min_true:
                m[s:e] = False

    x = m.astype(np.int8)
    boundaries = np.flatnonzero(np.r_[True, x[1:] != x[:-1], True])
    starts = boundaries[:-1]
    ends = boundaries[1:]
    vals = m[starts]

    if min_false > 1:
        for s, e, v in zip(starts, ends, vals):
            if (not v) and (e - s) < min_false:
                m[s:e] = True

    return m


def presence_mask_from_conf(
    conf: np.ndarray,
    p_thresh: float = 0.9,
    frac_min_parts: float = 0.4,
    smooth_win: int = 7,
    enter_thresh: float = 0.65,
    exit_thresh: float = 0.45,
    off_grace_frames: int = 6,
    min_on_frames: int = 10,
    min_off_frames: int = 6,
) -> np.ndarray:
    """
    Derives animal presence boolean mask from confidence matrix using
    hysteresis and smoothing filters.
    """
    T, K = conf.shape
    seen = (conf >= p_thresh).astype(float)
    frac = np.mean(seen, axis=1)

    kernel = np.ones(smooth_win) / smooth_win
    frac_smooth = np.convolve(frac, kernel, mode="same")

    mask = np.zeros(T, dtype=bool)
    in_pres = False
    bad_count = 0

    for t in range(T):
        f = frac_smooth[t]
        if not in_pres:
            if f >= enter_thresh and frac[t] >= frac_min_parts:
                in_pres = True
                bad_count = 0
                mask[t] = True
        else:
            if f < exit_thresh:
                bad_count += 1
                if bad_count >= off_grace_frames:
                    in_pres = False
                    mask[t] = False
                else:
                    mask[t] = True
            else:
                bad_count = 0
                mask[t] = True

    return clean_short_runs(mask, min_true=min_on_frames, min_false=min_off_frames)


def subset_by_activity(
    filtered_coordinates: Dict[str, np.ndarray],
    filtered_confidences: Dict[str, np.ndarray],
    video_frame_indexes: Dict[str, np.ndarray],
    subset_fraction: float = 0.1,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Selects contiguous windows containing highest cumulative body movement
    for each recording in a dataset.
    """
    coords_sub = {}
    confs_sub = {}
    vfi_sub = {}

    for k in filtered_coordinates.keys():
        c = filtered_coordinates[k]
        T = c.shape[0]
        window_size = max(1, int(round(T * subset_fraction)))

        if window_size >= T:
            coords_sub[k] = c
            confs_sub[k] = filtered_confidences[k]
            vfi_sub[k] = video_frame_indexes[k]
            continue

        disp = np.diff(c, axis=0)
        vel = np.linalg.norm(disp, axis=-1)
        mean_vel = np.nanmean(vel, axis=-1)

        rolling_disp = np.convolve(mean_vel, np.ones(window_size), mode="valid")
        best_start = int(np.argmax(rolling_disp))
        best_end = best_start + window_size

        coords_sub[k] = c[best_start:best_end]
        confs_sub[k] = filtered_confidences[k][best_start:best_end]
        vfi_sub[k] = video_frame_indexes[k][best_start:best_end]

    return coords_sub, confs_sub, vfi_sub

def smooth_heading_circular_legacy_fusion(theta: np.ndarray, window: int = 11) -> np.ndarray:
    """Preserve fusion's historical short-series and even-window behavior."""
    return smooth_heading_circular(theta, window, policy="legacy_fusion")


def smooth_centroid_median_legacy_fusion(c: np.ndarray, window: int = 11) -> np.ndarray:
    """Preserve fusion's historical short-series and even-window behavior."""
    return smooth_centroid_median(c, window, policy="legacy_fusion")


def presence_mask_from_conf_kpms(
    conf: np.ndarray,
    p_thresh: float = 0.9,
    frac_min_parts: float = 0.4,
    smooth_win: int = 7,
    enter_thresh: float = 0.65,
    exit_thresh: float = 0.45,
    off_grace_frames: int = 6,
    min_on_frames: int = 10,
    min_off_frames: int = 6,
) -> np.ndarray:
    """
    Generate boolean presence mask from bodypart confidence array (T, N).
    Uses hysteresis and smoothing to avoid rapid flapping on brief occlusions.
    """
    T, N = conf.shape

    valid = conf >= p_thresh
    count = valid.sum(axis=1)
    frac = count / float(N)

    if smooth_win > 1:
        k = np.ones(smooth_win, float) / smooth_win
        sm = np.convolve(frac, k, mode="same")
    else:
        sm = frac

    k_min = max(1, int(np.ceil(frac_min_parts * N)))
    seed = count >= k_min

    on = np.zeros(T, dtype=bool)
    inside = False
    below = 0
    for t in range(T):
        if not inside:
            if (sm[t] >= enter_thresh) or seed[t]:
                inside = True
                below = 0
        else:
            if (sm[t] <= exit_thresh) and (not seed[t]):
                below += 1
                if below >= off_grace_frames:
                    inside = False
                    below = 0
            else:
                below = 0
        on[t] = inside

    if min_on_frames > 1 or min_off_frames > 1:
        on = clean_short_runs(on, min_true=min_on_frames, min_false=min_off_frames, coerce_bool=False)

    return on
