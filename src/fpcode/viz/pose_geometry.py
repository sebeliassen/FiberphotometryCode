"""Visualization geometry; historical V2 behavior preserved."""

from __future__ import annotations

from fpcode.viz.hashing import sha256_file
from fpcode.viz.legacy_defaults import BOT_NOSE
from fpcode.viz.legacy_defaults import BOT_TAIL
from fpcode.viz.legacy_defaults import FUSED_DIR
from fpcode.viz.legacy_defaults import KPMS_FILTER_SIZE
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
import json
import math
import numpy as np


def filter_segment_pose(
    centroids: np.ndarray, headings: np.ndarray, filter_size: int = KPMS_FILTER_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter one COMPLETE continuous segment once, using the canonical KPMS
    algorithm (median filter on centroids; circular median on headings via
    the (cos, sin) representation in R^2).  Onset poses are then read from the
    filtered arrays.  Never filter separate short bout windows, and never run a
    linear median filter on wrapped angles.
    """
    from scipy.ndimage import median_filter

    centroids_arr = np.asarray(centroids, dtype=np.float64)
    headings_arr = np.asarray(headings, dtype=np.float64)

    # Median filter on centroids along time axis (axis 0)
    if centroids_arr.ndim == 1:
        c_filt = median_filter(centroids_arr, (int(filter_size),))
    else:
        c_filt = median_filter(centroids_arr, (int(filter_size), 1))

    # Circular median filter on headings via (sin, cos) in R^2
    sin_f = median_filter(np.sin(headings_arr), (int(filter_size),))
    cos_f = median_filter(np.cos(headings_arr), (int(filter_size),))
    h_filt = np.arctan2(sin_f, cos_f)

    return c_filt, h_filt


def egocentric_window(Y_window: np.ndarray, c0: np.ndarray, h0: float) -> np.ndarray:
    """
    Rigid ego-centric transform of ``(T, K, 2)`` coordinates about the onset
    centroid ``c0`` / heading ``h0``.  Algebraically identical to
    ``keypoint_moseq.util.inverse_rigid_transform`` (verified in tests); NaNs
    are preserved so confidence-aware comparisons can mask them.
    """
    d = np.asarray(Y_window, dtype=np.float64) - np.asarray(c0, dtype=np.float64)
    cos_h, sin_h = math.cos(h0), math.sin(h0)
    x = d[..., 0] * cos_h + d[..., 1] * sin_h
    y = -d[..., 0] * sin_h + d[..., 1] * cos_h
    return np.stack([x, y], axis=-1).astype(np.float32)


def active_bout_window(Y: np.ndarray, start: int, end: int) -> np.ndarray:
    """Extract only this execution. Never extend a short bout into another label."""
    if not 0 <= int(start) < int(end) <= len(Y):
        raise ValueError(f"Invalid active bout [{start}, {end}) for {len(Y)} frames")
    out = np.asarray(Y)[int(start) : int(end)]
    if not np.isfinite(out).all():
        raise ValueError(f"Nonfinite coordinates inside active bout [{start}, {end})")
    return out


def score_execution_counts(duration: int, pre: int, post: int) -> Dict[str, int]:
    """Fixed-onset score frames: prior context, this execution, later context."""
    active = min(int(duration), int(post))
    return {
        "prior_context_frames": int(pre),
        "active_execution_frames": active,
        "following_context_frames": int(post) - active,
        "score_window_frames": int(pre) + int(post),
    }


def shared_active_rms(
    a: np.ndarray, b: np.ndarray, duration_a: int, duration_b: int, pre: int, post: int, idxs: Sequence[int]
) -> Tuple[float, int]:
    """Diagnostic in native onset time; compare only the shared active interval."""
    n = min(int(duration_a), int(duration_b), int(post))
    if n <= 0:
        return float("nan"), 0
    return rms_point_difference(a[int(pre) : int(pre) + n], b[int(pre) : int(pre) + n], idxs), n


def fusion_geometry(base_sid: str, fused_dir: Path = FUSED_DIR) -> Dict[str, Any]:
    """Read the EXACT affine/reflection used to build the fused training artifact.

    Do not guess an epoch or reload a newly edited calibration. The existing
    fused artifact's recorded config is authoritative. Missing/stale metadata
    must be resolved on the workstation before canonical rendering is certified.
    """
    artifact = Path(fused_dir) / f"{base_sid}_tandem_21kp.h5"
    path = artifact.with_name(artifact.name + ".manifest.json")
    rec = json.loads(path.read_text())
    if rec.get("kind") != "fused_static_21kp":
        raise ValueError(f"Unexpected fused-artifact kind in {path}")
    st = artifact.stat()
    stored = rec.get("artifact", {})
    if (
        stored.get("name") != artifact.name
        or stored.get("size") != st.st_size
        or stored.get("mtime_ns") != st.st_mtime_ns
    ):
        raise ValueError(f"Fused artifact does not match its provenance: {path}")
    cfg = rec["config"]
    A = np.asarray(cfg["affine_matrix"], dtype=np.float64)
    b = np.asarray(cfg["affine_translation"], dtype=np.float64)
    W = float(cfg["arena_width"])
    if A.shape != (2, 2) or b.shape != (2,) or not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError(f"Invalid camera affine in {path}")
    if not math.isfinite(W) or W <= 0 or abs(float(np.linalg.det(A))) < 1e-12:
        raise ValueError(f"Invalid reflection width / singular affine in {path}")
    return {
        "A": A,
        "b": b,
        "reflection_width": W,
        "arena": cfg.get("arena"),
        "epoch": cfg.get("epoch"),
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
    }


def canonical_crop_matrices(
    c0: np.ndarray,
    h0: float,
    A: np.ndarray,
    b: np.ndarray,
    reflection_width: float,
    output_size: int = 160,
    top_window: float = 144.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Source->tile matrices, with exactly the SAME pose/geometry as scoring.

    Training uses p_top = A @ [W - q_bot.x, q_bot.y] + b.
    Therefore H_bot_to_top = [[A @ diag(-1,1), A @ [W,0] + b], [0,0,1]].
    M_top = scale * R(-h0) about c0; M_bot = M_top @ H_bot_to_top.
    Feed the 2x3 results to cv2.warpAffine WITHOUT WARP_INVERSE_MAP and WITHOUT
    a later cv2.flip. OpenCV performs the inverse raster lookup internally.
    No bottom nose/tail or paw-axis heading estimate is needed.
    """
    c0 = np.asarray(c0, dtype=np.float64)
    A, b = np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if c0.shape != (2,) or A.shape != (2, 2) or b.shape != (2,):
        raise ValueError("Expected c0(2,), A(2,2), b(2,)")
    if not np.isfinite(c0).all() or not math.isfinite(h0) or not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite camera geometry or onset pose")
    if top_window <= 0 or output_size <= 0 or not math.isfinite(reflection_width):
        raise ValueError("Invalid crop dimensions / reflection width")
    R = np.array([[math.cos(h0), math.sin(h0)], [-math.sin(h0), math.cos(h0)]])
    linear = (float(output_size) / float(top_window)) * R
    top = np.eye(3)
    top[:2, :2] = linear
    top[:2, 2] = float(output_size) / 2.0 - linear @ c0
    bot_to_top = np.eye(3)
    bot_to_top[:2, :2] = A @ np.diag([-1.0, 1.0])
    bot_to_top[:2, 2] = A @ np.array([float(reflection_width), 0.0]) + b
    return top[:2].astype(np.float32), (top @ bot_to_top)[:2].astype(np.float32)


def wrap_angle_deg(delta_rad: float) -> float:
    """Smallest signed angular difference in degrees."""
    return float(np.degrees(np.arctan2(np.sin(delta_rad), np.cos(delta_rad))))


def compute_robust_bot_pose(
    df_bot,
    bot_scorer,
    target_idx,
    search_window=10,
    top_heading_fallback: Optional[float] = None,
):
    """
    Return ``(centroid, heading, is_valid)`` for the BotCam frame ``target_idx``.

    Heading Resolution Cascade:
    1. BotCam Nose-to-Tail vector (if likelihoods >= 0.8 and 20 <= dist <= 400 px).
    2. Ventral Paw Longitudinal Axis (front-paws centroid - hind-paws centroid)
       if >= 1 front paw and >= 1 hind paw have likelihood >= 0.6 and separation >= 15 px.
    3. TopCam / fused body heading reflected across mirror parity:
       atan2(sin(h_top), -cos(h_top)).
    4. If all orientation methods fail, heading is returned as nan (caller flags as UNROTATED).

    Centroid is computed from nose/tail midpoint or paw centroid (if >= 2 paws with p >= 0.6).
    """
    n_frames = len(df_bot)
    target_idx = int(min(n_frames - 1, max(0, target_idx)))

    candidates = [target_idx]
    for w in range(1, search_window + 1):
        if target_idx - w >= 0:
            candidates.append(target_idx - w)
        if target_idx + w < n_frames:
            candidates.append(target_idx + w)

    for idx in candidates:
        try:
            nx = df_bot[bot_scorer, BOT_NOSE, "x"].iloc[idx]
            ny = df_bot[bot_scorer, BOT_NOSE, "y"].iloc[idx]
            np_ = df_bot[bot_scorer, BOT_NOSE, "likelihood"].iloc[idx]
            tx = df_bot[bot_scorer, BOT_TAIL, "x"].iloc[idx]
            ty = df_bot[bot_scorer, BOT_TAIL, "y"].iloc[idx]
            tp_ = df_bot[bot_scorer, BOT_TAIL, "likelihood"].iloc[idx]
            if np_ >= 0.8 and tp_ >= 0.8:
                dist = np.hypot(nx - tx, ny - ty)
                if 20.0 <= dist <= 400.0:
                    return ((nx + tx) / 2.0, (ny + ty) / 2.0), float(np.arctan2(ny - ty, nx - tx)), True
        except Exception:
            pass

    front_paws = ["left_front_paw", "right_front_paw"]
    hind_paws = ["left_hind_paw", "right_hind_paw"]
    try:
        fxs, fys = [], []
        for p in front_paws:
            px = df_bot[bot_scorer, p, "x"].iloc[target_idx]
            py = df_bot[bot_scorer, p, "y"].iloc[target_idx]
            pp = df_bot[bot_scorer, p, "likelihood"].iloc[target_idx]
            if np.isfinite(px) and np.isfinite(py) and pp >= 0.6:
                fxs.append(float(px))
                fys.append(float(py))
        hxs, hys = [], []
        for p in hind_paws:
            px = df_bot[bot_scorer, p, "x"].iloc[target_idx]
            py = df_bot[bot_scorer, p, "y"].iloc[target_idx]
            pp = df_bot[bot_scorer, p, "likelihood"].iloc[target_idx]
            if np.isfinite(px) and np.isfinite(py) and pp >= 0.6:
                hxs.append(float(px))
                hys.append(float(py))
        all_xs = fxs + hxs
        all_ys = fys + hys
        if len(all_xs) >= 2:
            c_bot = (float(np.mean(all_xs)), float(np.mean(all_ys)))
            # Tier 2: Paw longitudinal axis
            if len(fxs) >= 1 and len(hxs) >= 1:
                vx = float(np.mean(fxs) - np.mean(hxs))
                vy = float(np.mean(fys) - np.mean(hys))
                if np.hypot(vx, vy) >= 15.0:
                    return c_bot, float(np.arctan2(vy, vx)), True
            # Tier 3: TopCam heading fallback
            if top_heading_fallback is not None and np.isfinite(top_heading_fallback):
                h_ref = float(np.arctan2(np.sin(top_heading_fallback), -np.cos(top_heading_fallback)))
                return c_bot, h_ref, True
            return c_bot, float("nan"), True
    except Exception:
        pass

    # Fallback if insufficient paws: check if top heading fallback is available
    if top_heading_fallback is not None and np.isfinite(top_heading_fallback):
        h_ref = float(np.arctan2(np.sin(top_heading_fallback), -np.cos(top_heading_fallback)))
        return (486.0, 486.0), h_ref, False

    return (486.0, 486.0), float("nan"), False


def rms_point_difference(a: np.ndarray, b: np.ndarray, idxs: Sequence[int]) -> float:
    """
    RMS Euclidean point error in pixels between two ``(T, K, 2)`` windows over
    keypoints ``idxs``.  Used on the SCORED (preprocessed) representation,
    where both windows are fully finite by construction.
    """
    d = np.asarray(a, dtype=np.float64)[:, idxs] - np.asarray(b, dtype=np.float64)[:, idxs]
    return float(np.sqrt(np.mean(np.sum(d * d, axis=-1))))


def confidence_aware_rms(
    a: np.ndarray, b: np.ndarray, valid_a: np.ndarray, valid_b: np.ndarray, idxs: Sequence[int]
) -> Tuple[float, int]:
    """
    RMS point error over OBSERVED points only: a point contributes iff it is
    supported (finite x/y, likelihood >= threshold) in BOTH windows.  Missing
    or low-confidence points are excluded, never filled with zeros.  Returns
    ``(rms_px or nan, n_points_compared)``.
    """
    a = np.asarray(a, dtype=np.float64)[:, idxs]
    b = np.asarray(b, dtype=np.float64)[:, idxs]
    m = np.asarray(valid_a, dtype=bool)[:, idxs] & np.asarray(valid_b, dtype=bool)[:, idxs]
    m &= np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    d2 = np.sum((a - b) ** 2, axis=-1)
    return float(np.sqrt(d2[m].mean())), n
