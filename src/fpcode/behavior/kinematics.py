"""
fpcode.behavior.kinematics
==========================
Kinematic computation, heading and centroid smoothing,
activity window subsetting, and presence masking.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import median_filter


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
    sin_smooth = median_filter(np.sin(theta), size=window, mode="nearest")
    cos_smooth = median_filter(np.cos(theta), size=window, mode="nearest")
    return np.arctan2(sin_smooth, cos_smooth)


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
    return np.column_stack(
        [
            median_filter(c[:, 0], size=window, mode="nearest"),
            median_filter(c[:, 1], size=window, mode="nearest"),
        ]
    )


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
