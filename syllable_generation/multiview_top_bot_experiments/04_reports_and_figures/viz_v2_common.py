#!/usr/bin/env python3
"""
viz_v2_common.py
================
Single shared implementation for the Cohort Static 21kp V2 visualization chain
(selector -> grid-movie renderer -> trajectory renderer -> labeling gallery).

Why this module exists
----------------------
The 2026-10-07 density-sampling audit found that the same logic was
re-implemented (differently) in several scripts: the bout-end indexing was off
by one in two places, the renderer silently replaced selected bouts with
session-ordered substitutes, the timing windows were frame constants that
ignored the model FPS, and output caching ignored what was selected.  Every
repaired behavior now lives here and is covered by
`tests/test_v2_visualization_repairs.py`.

Design rules
------------
* Heavy imports (keypoint_moseq, jax, cv2, h5py) are done lazily inside the
  functions that need them so that the pure helpers are cheap to unit-test.
* Timing is configured in SECONDS and converted to frames from the model FPS.
* Bout bounds are canonical half-open ``[start, end)``.
* Eligibility (length, window, finite pose, original-confidence QC) is decided
  BEFORE tiers/selection.  Renderers never choose different bouts.
* Every output carries a provenance sidecar; cached output is reused only when
  the sidecar's cache key equals the current inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# =============================================================================
# Paths and constants
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXP_DIR.parent.parent

COHORT_DIR = EXP_DIR / "cohort_static_21kp_v2"
MODEL_NAME = "model_21kp_static_50ar_300slds_seed101"
MODEL_DIR = COHORT_DIR / MODEL_NAME
RESULTS_PATH = MODEL_DIR / "results.h5"
CONFIG_PATH = COHORT_DIR / "config.yml"
VFI_PATH = COHORT_DIR / "video_frame_indexes.npz"
PREPROCESSED_PATH = COHORT_DIR / "preprocessed_data.pkl"
# Fused tracks carry the RAW DeepLabCut likelihoods verbatim (see
# core/static_fusion_engine.py, "NO anchor-confidence coupling") and the
# bilateral PAW_MAP swap + epoch affine are already applied there.  Row i of
# a fused file is TopCam frame i, which is exactly what the VFI stores.
FUSED_DIR = EXP_DIR / "data/cohort_fused_static_21kp_v2"

TOP_VID_DIR = REPO_ROOT / "karen_project/video_files"
BOT_VID_DIR = REPO_ROOT / "karen_project/botcam_video_files"

# Fresh, versioned output directory (the pre-repair gallery stays untouched).
VIZ_VERSION = "v4"
DEFAULT_VIZ_DIR = SCRIPT_DIR / f"visualizations_cohort_static_21kp_v2_seed101_{VIZ_VERSION}"

DORSAL_BODYPARTS = [
    "nose", "left_ear_tip", "right_ear_tip", "neck", "mid_back",
    "mouse_center", "mid_backend", "mid_backend2", "mid_backend3",
    "tail_base", "left_shoulder", "left_midside", "left_hip",
    "right_shoulder", "right_midside", "right_hip", "head_midpoint",
]
PAW_BODYPARTS = ["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw"]
BODYPARTS_21KP = DORSAL_BODYPARTS + PAW_BODYPARTS
DORSAL_IDXS = list(range(len(DORSAL_BODYPARTS)))
PAW_IDXS = list(range(len(DORSAL_BODYPARTS), len(BODYPARTS_21KP)))
FRONT_PAW_BODYPARTS = ["left_front_paw", "right_front_paw"]
HIND_PAW_BODYPARTS = ["left_hind_paw", "right_hind_paw"]
FRONT_PAW_IDXS = [BODYPARTS_21KP.index(bp) for bp in FRONT_PAW_BODYPARTS]
HIND_PAW_IDXS = [BODYPARTS_21KP.index(bp) for bp in HIND_PAW_BODYPARTS]

# Actual BotCam column names (the pre-repair fallback used paw_front_left etc.,
# which do not exist in the DLC files).
BOT_NOSE = "nose_tip"
BOT_TAIL = "tail_base"
BOT_PAW_NAMES = ("left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw")

CHAMBER_TO_ARENA = {
    "top_left": "Arena1",
    "bottom_left": "Arena2",
    "bottom_right": "Arena3",
    "top_right": "Arena4",
}

SCHEMA_VERSION = 4

# Density settings are the installed KPMS defaults; they are recorded, never
# silently changed (pca_dim, n_neighbors, pca_samples).
DENSITY_PCA_DIM = 4
DENSITY_N_NEIGHBORS = 50
DENSITY_PCA_SAMPLES = 50000
KPMS_FILTER_SIZE = 9  # keypoint_moseq.util.get_instance_trajectories default

# Old KPMS defaults were pre=5 / post=15 at 30 fps.  The V2 model runs at 60 fps,
# so the same ~167 ms : 500 ms proportions are 10 + 30 frames (40 total).
DEFAULT_SCORE_PRE_SECONDS = 1.0 / 6.0
DEFAULT_SCORE_POST_SECONDS = 0.5
DEFAULT_MOVIE_PRE_SECONDS = 1.0
DEFAULT_MOVIE_POST_SECONDS = 2.0
DEFAULT_MIN_BOUT_SECONDS = 5.0 / 60.0  # the previous 5-frame minimum at 60 fps


# =============================================================================
# Timing: seconds are the configuration, frames are derived from FPS
# =============================================================================
def frames_from_seconds(seconds: float, fps: float) -> int:
    """Convert a duration in seconds to a whole number of frames at ``fps``."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return int(round(float(seconds) * float(fps)))


def make_window(fps: float, pre_seconds: float, post_seconds: float) -> Dict[str, Any]:
    """Build a timing window record storing both seconds and derived frames."""
    pre_frames = frames_from_seconds(pre_seconds, fps)
    post_frames = frames_from_seconds(post_seconds, fps)
    if pre_frames < 0 or post_frames <= 0:
        raise ValueError(f"Degenerate window: pre={pre_frames}f post={post_frames}f at {fps} fps")
    return {
        "fps": float(fps),
        "pre_seconds": float(pre_seconds),
        "post_seconds": float(post_seconds),
        "pre_frames": int(pre_frames),
        "post_frames": int(post_frames),
        "total_frames": int(pre_frames + post_frames),
        "total_seconds_effective": float((pre_frames + post_frames) / fps),
    }


# =============================================================================
# Canonical half-open bout bounds
# =============================================================================
def half_open_bouts(z: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return ``(starts, ends)`` of contiguous runs of equal labels as half-open
    ``[start, end)`` bounds.  ``end - start`` is the true duration in frames.

    An empty sequence yields two empty arrays (no bouts).
    """
    z = np.asarray(z)
    if z.ndim != 1:
        raise ValueError(f"label sequence must be 1-D, got shape {z.shape}")
    n = len(z)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    changes = np.flatnonzero(z[1:] != z[:-1]) + 1
    starts = np.r_[0, changes].astype(np.int64)
    ends = np.r_[changes, n].astype(np.int64)
    return starts, ends


# =============================================================================
# Identity and deterministic ordering
# =============================================================================
def make_bout_id(segment_name: str, onset: int) -> str:
    """Stable, unique bout identifier (a label can start once per segment frame)."""
    return f"{segment_name}:{int(onset)}"


def stable_ranks(keys: Sequence[Tuple[str, int]]) -> np.ndarray:
    """
    Rank of every (segment_name, onset) key in lexicographic order.  Used to
    break exact distance ties deterministically (stable bout-ID order).
    """
    order = sorted(range(len(keys)), key=lambda i: (keys[i][0], int(keys[i][1])))
    ranks = np.empty(len(keys), dtype=np.int64)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def base_session_id(segment_name: str) -> str:
    """Strip the ``_segNNN`` suffix added when sessions are split at ventral gaps."""
    return re.sub(r"_seg\d+$", "", segment_name)


def get_session_files(sid: str, pairs_by_sid: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Resolve Top/Bot videos and DLC h5 files for a (possibly segmented) session id."""
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})_([a-z]+_[a-z]+)", sid)
    if not m:
        return None
    ts, chamber = m.group(1), m.group(2)
    arena = CHAMBER_TO_ARENA.get(chamber)
    if not arena:
        return None
    pair = pairs_by_sid.get(f"{ts}_{chamber}")
    if not pair:
        return None

    top_vids = list(TOP_VID_DIR.glob(f"*{ts}*{chamber}*.avi"))
    if len(top_vids) > 1 and "top_h5" in pair:
        top_stem = re.sub(r"DLC_.*\.h5$", "", Path(pair["top_h5"]).name)
        matched = [v for v in top_vids if v.stem == top_stem or top_stem in v.name]
        if matched:
            top_vids = matched[:1]

    bot_vids = list(BOT_VID_DIR.glob(f"*{arena}*BotCamVid*{ts}*.avi"))
    if len(bot_vids) > 1 and "top_h5" in pair:
        has_two = " (2)" in Path(pair["top_h5"]).name
        matched = [v for v in bot_vids if (" (2)" in v.name) == has_two]
        if matched:
            bot_vids = matched[:1]
        else:
            bot_vids = bot_vids[:1]

    if len(top_vids) > 1 or len(bot_vids) > 1:
        raise ValueError(f"Ambiguous source videos for {sid}: top={top_vids}, bot={bot_vids}")
    if top_vids and bot_vids and Path(pair["top_h5"]).exists() and Path(pair["bot_h5"]).exists():
        return {
            "top_vid": str(top_vids[0]),
            "bot_vid": str(bot_vids[0]),
            "top_h5": str(pair["top_h5"]),
            "bot_h5": str(pair["bot_h5"]),
        }
    return None


# =============================================================================
# Pose filtering and egocentric alignment (shared by scoring, plots, crops)
# =============================================================================
def filter_segment_pose(centroids: np.ndarray, headings: np.ndarray,
                        filter_size: int = KPMS_FILTER_SIZE) -> Tuple[np.ndarray, np.ndarray]:
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


def movie_context_ok(start: int, n_frames: int, pre: int, post: int) -> bool:
    """True iff the complete half-open context [start-pre, start+post) exists."""
    return int(start) >= int(pre) and int(start) + int(post) <= int(n_frames)


def require_fused_qc(fused_data, base_sid: str, skip: bool = False):
    """Missing tracking evidence is an error, never an implicit QC pass."""
    if fused_data is None and not skip:
        raise RuntimeError(f"Original fused tracks unavailable for {base_sid}; QC cannot be certified")
    return fused_data


def active_bout_window(Y: np.ndarray, start: int, end: int) -> np.ndarray:
    """Extract only this execution. Never extend a short bout into another label."""
    if not 0 <= int(start) < int(end) <= len(Y):
        raise ValueError(f"Invalid active bout [{start}, {end}) for {len(Y)} frames")
    out = np.asarray(Y)[int(start):int(end)]
    if not np.isfinite(out).all():
        raise ValueError(f"Nonfinite coordinates inside active bout [{start}, {end})")
    return out


def score_execution_counts(duration: int, pre: int, post: int) -> Dict[str, int]:
    """Fixed-onset score frames: prior context, this execution, later context."""
    active = min(int(duration), int(post))
    return {"prior_context_frames": int(pre), "active_execution_frames": active,
            "following_context_frames": int(post) - active,
            "score_window_frames": int(pre) + int(post)}


def shared_active_rms(a: np.ndarray, b: np.ndarray, duration_a: int, duration_b: int,
                      pre: int, post: int, idxs: Sequence[int]) -> Tuple[float, int]:
    """Diagnostic in native onset time; compare only the shared active interval."""
    n = min(int(duration_a), int(duration_b), int(post))
    if n <= 0:
        return float("nan"), 0
    return rms_point_difference(a[int(pre):int(pre)+n], b[int(pre):int(pre)+n], idxs), n


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
    if stored.get("name") != artifact.name or stored.get("size") != st.st_size or stored.get("mtime_ns") != st.st_mtime_ns:
        raise ValueError(f"Fused artifact does not match its provenance: {path}")
    cfg = rec["config"]
    A = np.asarray(cfg["affine_matrix"], dtype=np.float64)
    b = np.asarray(cfg["affine_translation"], dtype=np.float64)
    W = float(cfg["arena_width"])
    if A.shape != (2, 2) or b.shape != (2,) or not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError(f"Invalid camera affine in {path}")
    if not math.isfinite(W) or W <= 0 or abs(float(np.linalg.det(A))) < 1e-12:
        raise ValueError(f"Invalid reflection width / singular affine in {path}")
    return {"A": A, "b": b, "reflection_width": W, "arena": cfg.get("arena"),
            "epoch": cfg.get("epoch"), "manifest_path": str(path), "manifest_sha256": sha256_file(path)}


def canonical_crop_matrices(c0: np.ndarray, h0: float, A: np.ndarray, b: np.ndarray,
                            reflection_width: float, output_size: int = 160,
                            top_window: float = 144.0) -> Tuple[np.ndarray, np.ndarray]:
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
    if c0.shape != (2,) or A.shape != (2,2) or b.shape != (2,):
        raise ValueError("Expected c0(2,), A(2,2), b(2,)")
    if not np.isfinite(c0).all() or not math.isfinite(h0) or not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite camera geometry or onset pose")
    if top_window <= 0 or output_size <= 0 or not math.isfinite(reflection_width):
        raise ValueError("Invalid crop dimensions / reflection width")
    R = np.array([[math.cos(h0), math.sin(h0)], [-math.sin(h0), math.cos(h0)]])
    linear = (float(output_size) / float(top_window)) * R
    top = np.eye(3)
    top[:2,:2] = linear
    top[:2,2] = float(output_size) / 2.0 - linear @ c0
    bot_to_top = np.eye(3)
    bot_to_top[:2,:2] = A @ np.diag([-1., 1.])
    bot_to_top[:2,2] = A @ np.array([float(reflection_width), 0.]) + b
    return top[:2].astype(np.float32), (top @ bot_to_top)[:2].astype(np.float32)


def source_input_hashes(cache) -> Dict[str, str]:
    return {"results_sha256": cache.get(RESULTS_PATH), "config_sha256": cache.get(CONFIG_PATH),
            "vfi_sha256": cache.get(VFI_PATH), "preprocessed_sha256": cache.get(PREPROCESSED_PATH)}


def verify_manifest_inputs(manifest: Dict[str, Any], cache) -> None:
    """Reject selection provenance tied to different model/coordinate arrays."""
    expected = manifest.get("input_hashes")
    if not expected or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Regenerate a schema-v4 selection manifest before rendering")
    if not manifest.get("qc_config",{}).get("applied",False):
        raise ValueError("QC was bypassed; this debug manifest cannot certify a production gallery")
    current = source_input_hashes(cache)
    changed = [k for k,v in current.items() if expected.get(k) != v]
    if changed:
        raise ValueError("Selection input mismatch: " + ", ".join(changed))


def media_file_identity(manifest: Dict[str, Any], syllable: int) -> Dict[str, Any]:
    """Cache invalidation for selected source media/tracks (path/size/mtime)."""
    paths = set()
    for b in manifest["syllables"][str(syllable)]["bouts"]:
        if b.get("is_blank"):
            continue
        paths.update(str(b[k]) for k in ("top_vid", "bot_vid", "top_h5", "bot_h5"))
        fused = FUSED_DIR / f"{base_session_id(b['segment_name'])}_tandem_21kp.h5"
        paths.update((str(fused), str(fused.with_name(fused.name+'.manifest.json'))))
    result = {}
    for name in sorted(paths):
        p = Path(name)
        st = p.stat()
        result[name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    return result


def selection_provenance_matches(sidecar: Path, manifest: Dict[str, Any], syllable: int,
                                 code_files: Sequence[Path] = ()) -> bool:
    """Gallery validation checks actual IDs/status, rather than sidecar existence."""
    try:
        rec=json.loads(Path(sidecar).read_text())
        key=rec["cache_key"]
        if key.get("schema_version")!=SCHEMA_VERSION or key.get("manifest_syllable_sha256")!=manifest_syllable_hash(manifest,syllable) or key.get("manifest_global_sha256")!=manifest_global_hash(manifest):
            return False
        if code_files and key.get("code_sha256")!=hash_files(code_files):
            return False
        if code_files and (key.get("selected_source_file_identity")!=media_file_identity(manifest,syllable)
                           or key.get("calibration_sha256")!=hash_files(CALIBRATION_FILES)):
            return False
        wanted=selection_ids(manifest,syllable)
        if not wanted:
            return False
        if "rendered_tiles" in rec:
            tiles=rec["rendered_tiles"]
            if any(t.get("flagged_invalid") for t in tiles):
                return False
            actual=[t.get("bout_id") for t in tiles if not t.get("is_blank")]
        else:
            actual=rec["rendered_bouts"]
        return actual==wanted and len(actual)==len(set(actual))
    except (OSError,ValueError,KeyError,TypeError):
        return False


def wrap_angle_deg(delta_rad: float) -> float:
    """Smallest signed angular difference in degrees."""
    return float(np.degrees(np.arctan2(np.sin(delta_rad), np.cos(delta_rad))))


# =============================================================================
# Original-confidence tracking QC (scored interval + active execution)
# =============================================================================
@dataclass(frozen=True)
class QCConfig:
    """
    Quality rules applied to ORIGINAL DLC likelihood / missingness (fused tracks
    keep raw likelihoods).  ``conf_threshold`` defaults to the cohort config's
    ``conf_threshold``.  The fraction/gap thresholds are an explicit,
    reported starting point (NOT a validated optimum): they avoid demanding
    that every landmark be perfect in every frame while rejecting executions
    whose paws or dorsal points are mostly unsupported.  Coverage under
    alternative thresholds is written to ``qc_threshold_sensitivity.json``.

    The canonical reference policy is "strict_all_paws".
    An optional opt-in policy "dorsal_and_hind_supported_forepaw_limited"
    relaxes forepaw support requirements when dorsal and both hindpaws pass.
    """
    conf_threshold: float = 0.5
    min_dorsal_valid_fraction: float = 0.8
    max_dorsal_gap_seconds: float = 0.1
    min_paw_valid_fraction: float = 0.8
    max_paw_gap_seconds: float = 0.1
    policy: str = "strict_all_paws"
    min_forepaw_valid_fraction: float = 0.0
    max_forepaw_gap_seconds: float = float("inf")


def valid_track_mask(xy: np.ndarray, conf: np.ndarray, conf_threshold: float) -> np.ndarray:
    """A point is supported iff x AND y are finite and its likelihood is >= threshold."""
    xy = np.asarray(xy)
    conf = np.asarray(conf)
    finite_xy = np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])
    with np.errstate(invalid="ignore"):
        confident = np.isfinite(conf) & (conf >= conf_threshold)
    return finite_xy & confident


def longest_true_run(mask: np.ndarray) -> np.ndarray:
    """Longest run of True along axis 0 for every column of a ``(W, K)`` mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 1:
        mask = mask[:, None]
    W = mask.shape[0]
    if W == 0:
        return np.zeros(mask.shape[1], dtype=np.int64)
    idx = np.arange(W)[:, None]
    last_false = np.maximum.accumulate(np.where(~mask, idx, -1), axis=0)
    run = np.where(mask, idx - last_false, 0)
    return run.max(axis=0).astype(np.int64)


def window_frac_gap(valid_window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Per-keypoint supported fraction and longest unsupported gap over a ``(W, 21)`` mask."""
    valid_window = np.asarray(valid_window, dtype=bool)
    W = valid_window.shape[0]
    if W == 0:
        return np.zeros(valid_window.shape[1]), np.zeros(valid_window.shape[1], dtype=np.int64), 0
    return valid_window.mean(axis=0), longest_true_run(~valid_window), W


def summarize_window_qc(valid_window: np.ndarray) -> Dict[str, Any]:
    """
    Per-group QC summary of a ``(W, 21)`` supported-point mask.  Reports
    per-paw valid fractions / maximum unsupported gaps so any rejection can be
    traced to a specific landmark.
    """
    frac, gap, W = window_frac_gap(valid_window)
    return summarize_frac_gap(frac, gap, W)


def summarize_frac_gap(frac: np.ndarray, gap: np.ndarray, W: int) -> Dict[str, Any]:
    """Same summary as ``summarize_window_qc`` from precomputed (fraction, gap) vectors."""
    frac = np.asarray(frac, dtype=np.float64)
    gap = np.asarray(gap)
    dorsal_frac, dorsal_gap = frac[DORSAL_IDXS], gap[DORSAL_IDXS]
    paw_frac, paw_gap = frac[PAW_IDXS], gap[PAW_IDXS]
    front_frac, front_gap = frac[FRONT_PAW_IDXS], gap[FRONT_PAW_IDXS]
    hind_frac, hind_gap = frac[HIND_PAW_IDXS], gap[HIND_PAW_IDXS]
    return {
        "window_frames": int(W),
        "min_dorsal_valid_fraction": float(dorsal_frac.min()),
        "max_dorsal_gap_frames": int(dorsal_gap.max()),
        "min_paw_valid_fraction": float(paw_frac.min()),
        "max_paw_gap_frames": int(paw_gap.max()),
        "min_front_paw_valid_fraction": float(front_frac.min()) if len(front_frac) else 0.0,
        "max_front_paw_gap_frames": int(front_gap.max()) if len(front_gap) else 0,
        "min_hind_paw_valid_fraction": float(hind_frac.min()) if len(hind_frac) else 0.0,
        "max_hind_paw_gap_frames": int(hind_gap.max()) if len(hind_gap) else 0,
        "per_paw_valid_fraction": {n: float(f) for n, f in zip(PAW_BODYPARTS, paw_frac)},
        "per_paw_max_gap_frames": {n: int(g) for n, g in zip(PAW_BODYPARTS, paw_gap)},
    }


def qc_decision(summary: Dict[str, Any], cfg: QCConfig, fps: float) -> Tuple[bool, List[str]]:
    """
    Apply the QC rules to a window summary.  Only the dorsal and PAW columns of
    the fused tracks are consulted: good BotCam nose/tail anchors can never
    certify wholly missing paws.
    """
    reasons: List[str] = []
    dorsal_gap_limit = frames_from_seconds(cfg.max_dorsal_gap_seconds, fps)
    paw_gap_limit = frames_from_seconds(cfg.max_paw_gap_seconds, fps)
    if summary["min_dorsal_valid_fraction"] < cfg.min_dorsal_valid_fraction:
        reasons.append("qc_dorsal_fraction")
    if summary["max_dorsal_gap_frames"] > dorsal_gap_limit:
        reasons.append("qc_dorsal_gap")

    if cfg.policy == "dorsal_and_hind_supported_forepaw_limited":
        hind_frac = summary.get("min_hind_paw_valid_fraction", summary["min_paw_valid_fraction"])
        hind_gap = summary.get("max_hind_paw_gap_frames", summary["max_paw_gap_frames"])
        if hind_frac < cfg.min_paw_valid_fraction:
            reasons.append("qc_hind_paw_fraction")
        if hind_gap > paw_gap_limit:
            reasons.append("qc_hind_paw_gap")

        fore_frac = summary.get("min_front_paw_valid_fraction", 1.0)
        fore_gap = summary.get("max_front_paw_gap_frames", 0)
        fore_gap_limit = (
            frames_from_seconds(cfg.max_forepaw_gap_seconds, fps)
            if np.isfinite(cfg.max_forepaw_gap_seconds)
            else float("inf")
        )
        if fore_frac < cfg.min_forepaw_valid_fraction:
            reasons.append("qc_forepaw_fraction")
        if fore_gap > fore_gap_limit:
            reasons.append("qc_forepaw_gap")
    else:
        if summary["min_paw_valid_fraction"] < cfg.min_paw_valid_fraction:
            reasons.append("qc_paw_fraction")
        if summary["max_paw_gap_frames"] > paw_gap_limit:
            reasons.append("qc_paw_gap")

    return (len(reasons) == 0), reasons


def collapse_qc_reason(reasons: Iterable[str]) -> str:
    """Single rejection bucket (dorsal wins over paws) for count reporting."""
    reasons = list(reasons)
    if any(r.startswith("qc_dorsal") for r in reasons):
        return "qc_dorsal"
    if any("hind" in r for r in reasons):
        return "qc_hind_paws"
    if any("fore" in r for r in reasons):
        return "qc_fore_paws"
    return "qc_paws"


def load_fused_tracks(base_sid: str, fused_dir: Path = FUSED_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the ORIGINAL fused observations for a session: ``xy (T, 21, 2)`` and
    raw likelihoods ``conf (T, 21)``, in ``BODYPARTS_21KP`` order.  Missing
    BotCam-mapped frames are NaN / likelihood 0 (see static_fusion_engine).
    """
    import pandas as pd

    path = Path(fused_dir) / f"{base_sid}_tandem_21kp.h5"
    df = pd.read_hdf(path)
    scorer = df.columns.levels[0][0]
    x = np.column_stack([df[(scorer, bp, "x")].to_numpy(dtype=np.float32) for bp in BODYPARTS_21KP])
    y = np.column_stack([df[(scorer, bp, "y")].to_numpy(dtype=np.float32) for bp in BODYPARTS_21KP])
    conf = np.column_stack([df[(scorer, bp, "likelihood")].to_numpy(dtype=np.float32) for bp in BODYPARTS_21KP])
    return np.stack([x, y], axis=-1), conf


# =============================================================================
# Tiered selection (density ratio -> compact neighborhood -> all bouts)
# =============================================================================
def choose_tier(n_eligible: int, G: int) -> str:
    """Tier from the ELIGIBLE count (determined after QC, never before)."""
    if n_eligible <= 0:
        return "no_eligible_bouts"
    if n_eligible > DENSITY_N_NEIGHBORS:
        return "tier_1_density"
    if n_eligible > G:
        return "tier_2_compact_neighborhood"
    return "tier_3_all_bouts"


def density_ratio_scores(X_pca: np.ndarray, global_nbrs, n_neighbors: int = DENSITY_N_NEIGHBORS):
    """
    Installed-KPMS density ratio (keypoint_moseq.util.sample_instances):

        local_density  = 1 / mean(kNN distances within the syllable, self included)
        global_density = 1 / mean(kNN distances to the background fit, self not guaranteed)
        ratio          = local_density / global_density

    The self-neighbor convention is KPMS's: the query point is part of the
    syllable's own fit, so its first neighbor is itself at distance 0.  A zero
    mean distance (all points identical) is floored to avoid a division error;
    KPMS itself has no floor, so any difference only appears in that degenerate
    case.
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_pca)
    distances, indices = nbrs.kneighbors(X_pca)
    local_mean = np.maximum(distances.mean(axis=1), np.finfo(np.float64).tiny)
    g_dist, _ = global_nbrs.kneighbors(X_pca)
    global_mean = np.maximum(g_dist.mean(axis=1), np.finfo(np.float64).tiny)
    ratio = (1.0 / local_mean) / (1.0 / global_mean)
    return ratio, distances, indices, 1.0 / local_mean, 1.0 / global_mean


def order_group_exemplar_first(exemplar: int, cand_idx: Sequence[int], cand_dist: Sequence[float],
                               stable_rank: np.ndarray, G: int) -> List[int]:
    """
    The display group: the scored exemplar FIRST, then its nearest candidates in
    increasing distance; exactly equal distances are ordered by stable bout ID.
    The exemplar is inserted explicitly (a tie in the KNN query can otherwise
    omit it) and removed from the remaining candidates (no duplicate).
    """
    others = [(float(d), int(stable_rank[i]), int(i)) for i, d in zip(cand_idx, cand_dist) if int(i) != int(exemplar)]
    others.sort()
    return [int(exemplar)] + [i for _, _, i in others][: max(0, G - 1)]


def select_tier1_density(X_pca: np.ndarray, global_nbrs, stable_rank: np.ndarray, G: int,
                         n_neighbors: int = DENSITY_N_NEIGHBORS) -> Dict[str, Any]:
    """Tier 1 (N > 50): KPMS density-ratio exemplar, then nearest G around it."""
    ratio, distances, indices, local_d, global_d = density_ratio_scores(X_pca, global_nbrs, n_neighbors)
    exemplar = int(np.argmax(ratio))
    nb_idx = [int(i) for i in indices[exemplar]]
    nb_dist = [float(d) for d in distances[exemplar]]
    chosen = order_group_exemplar_first(exemplar, nb_idx, nb_dist, stable_rank, G)
    return {
        "anchor": exemplar,
        "chosen": chosen,
        "neighbor_idx": nb_idx,
        "neighbor_dist": nb_dist,
        "density_ratio": float(ratio[exemplar]),
        "local_density": float(local_d[exemplar]),
        "global_density": float(global_d[exemplar]),
    }


def _medoid_and_order(D: np.ndarray, members: Sequence[int], stable_rank: np.ndarray) -> List[int]:
    """Medoid of ``members`` first, then the rest by (distance to medoid, stable ID)."""
    members = [int(m) for m in members]
    sub = D[np.ix_(members, members)]
    sums = sub.sum(axis=1)
    best = min(range(len(members)), key=lambda k: (float(sums[k]), int(stable_rank[members[k]])))
    medoid = members[best]
    rest = sorted((m for m in members if m != medoid), key=lambda m: (float(D[medoid, m]), int(stable_rank[m])))
    return [medoid] + rest


def select_tier2_compact(trajs: np.ndarray, stable_rank: np.ndarray, G: int) -> Dict[str, Any]:
    """Tier 2 (G < N <= 50): candidate sets of each bout + its G-1 nearest; lowest mean pairwise distance wins."""
    from sklearn.metrics import pairwise_distances

    N = len(trajs)
    D = pairwise_distances(trajs)
    best_score, best_set = None, None
    for i in range(N):
        order = sorted((j for j in range(N) if j != i), key=lambda j: (float(D[i, j]), int(stable_rank[j])))
        cand = [i] + order[: G - 1]
        sub = D[np.ix_(cand, cand)]
        score = float(sub[np.triu_indices(G, k=1)].mean())
        if best_score is None or score < best_score - 1e-12:
            best_score, best_set = score, cand
    chosen = _medoid_and_order(D, best_set, stable_rank)
    return {"anchor": chosen[0], "chosen": chosen, "compactness_score": best_score}


def select_tier3_all(trajs: np.ndarray, stable_rank: np.ndarray) -> Dict[str, Any]:
    """Tier 3 (N <= G): every eligible bout once, medoid first; blanks are added by the caller."""
    N = len(trajs)
    if N == 1:
        return {"anchor": 0, "chosen": [0]}
    from sklearn.metrics import pairwise_distances

    D = pairwise_distances(trajs)
    chosen = _medoid_and_order(D, list(range(N)), stable_rank)
    return {"anchor": chosen[0], "chosen": chosen}


# =============================================================================
# Shared final selection: one list drives movies AND trajectory plots
# =============================================================================
def assert_unique_bout_ids(bouts: Iterable[Dict[str, Any]], context: str = "") -> None:
    ids = [b["bout_id"] for b in bouts if not b.get("is_blank", False)]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise AssertionError(f"Duplicate bout IDs in final selection{(' (' + context + ')') if context else ''}: {dup}")


def selection_ids(manifest: Dict[str, Any], syllable: int) -> List[str]:
    """Ordered non-blank bout IDs of the finalized manifest (the only source of truth)."""
    entry = manifest["syllables"][str(int(syllable))]
    return [b["bout_id"] for b in entry["bouts"] if not b.get("is_blank", False)]


def resolve_display_bouts(manifest_bouts: Sequence[Dict[str, Any]],
                          target_validator: Callable[[Dict[str, Any]], Tuple[bool, str]],
                          log: Callable[[str], None] = lambda msg: None) -> List[Dict[str, Any]]:
    """
    Turn the finalized manifest list into display tiles WITHOUT choosing
    different executions.  A bout whose target execution unexpectedly fails
    validation becomes an explicitly flagged blank that keeps the original
    bout ID; it is never replaced by another bout, and no flag (exemplar,
    medoid) can transfer to anything else.
    """
    tiles: List[Dict[str, Any]] = []
    for pos, b in enumerate(manifest_bouts):
        if b.get("is_blank", False):
            tile = dict(b)
            tile["tile_index"] = pos
            tile["render_status"] = "blank_in_manifest"
            tiles.append(tile)
            continue
        ok, reason = target_validator(b)
        if ok:
            tile = dict(b)
            tile["tile_index"] = pos
            tile["is_blank"] = False
            tile["flagged_invalid"] = False
            tile["render_status"] = "rendered"
            tiles.append(tile)
        else:
            log(f"  [FLAGGED BLANK] tile {pos} bout {b['bout_id']} failed target validation: {reason}")
            tiles.append({
                "is_blank": True,
                "flagged_invalid": True,
                "bout_id": b["bout_id"],
                "invalid_reason": reason,
                "is_anchor": bool(b.get("is_anchor", False)),
                "anchor_type": b.get("anchor_type"),
                "tile_index": pos,
                "render_status": "flagged_invalid",
            })
    return tiles


def assert_rendered_equals_manifest(tiles: Sequence[Dict[str, Any]],
                                    manifest_bouts: Sequence[Dict[str, Any]]) -> None:
    """The displayed ID sequence must equal the finalized manifest sequence, in order."""
    got = [t.get("bout_id") for t in tiles if t.get("render_status") in ("rendered", "flagged_invalid")]
    want = [b["bout_id"] for b in manifest_bouts if not b.get("is_blank", False)]
    if got != want:
        raise AssertionError(f"Rendered IDs differ from finalized manifest.\n  rendered: {got}\n  manifest: {want}")
    if len(set(got)) != len(got):
        raise AssertionError(f"Duplicate rendered bout IDs: {got}")
    assert_unique_bout_ids(manifest_bouts, "manifest")


def grid_layout(G: int) -> Tuple[int, int]:
    """(rows, cols) for ``G`` displayed slots.  12 -> 3x4, 24 -> 4x6 (the KPMS default)."""
    known = {12: (3, 4), 24: (4, 6)}
    if G in known:
        return known[G]
    cols = int(math.ceil(math.sqrt(G * 4.0 / 3.0)))
    rows = int(math.ceil(G / cols))
    return rows, cols


# =============================================================================
# BotCam pose helper (fixed paw names, no arbitrary zero-heading fallback)
# =============================================================================
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


# =============================================================================
# Provenance: hashes, sidecars, cache decisions
# =============================================================================
def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401
        return _json_default(o)


class HashCache:
    """
    sha256 of large files (e.g. results.h5), memoized by (size, mtime_ns) in a
    small JSON file so each renderer does not re-read hundreds of MB.
    """

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        try:
            self._d = json.loads(self.cache_path.read_text())
        except Exception:
            self._d = {}

    def get(self, path: Path) -> str:
        path = Path(path)
        st = path.stat()
        key = str(path.resolve())
        rec = self._d.get(key)
        if rec and rec.get("size") == st.st_size and rec.get("mtime_ns") == st.st_mtime_ns:
            return rec["sha256"]
        digest = sha256_file(path)
        self._d[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._d, indent=1))
        os.replace(tmp, self.cache_path)
        return digest


CALIBRATION_FILES = [
    EXP_DIR / "core/camera_correspondence.py",
    EXP_DIR / "core/pair_matching.py",
    EXP_DIR / "core/ttl_sync.py",
    EXP_DIR / "core/static_fusion_engine.py",
    EXP_DIR / "core/epoch_arena_transforms.json",
    EXP_DIR / "core/static_arena_transforms.json",
    EXP_DIR / "core/ttl_cohort_offsets.json",
]


def hash_files(paths: Iterable[Path]) -> Dict[str, str]:
    out = {}
    for p in paths:
        p = Path(p)
        out[p.name] = sha256_file(p) if p.exists() else "MISSING"
    return out


def manifest_syllable_hash(manifest: Dict[str, Any], syllable: int) -> str:
    return sha256_json(manifest["syllables"][str(int(syllable))])


def manifest_global_hash(manifest: Dict[str, Any]) -> str:
    return sha256_json({k: v for k, v in manifest.items() if k != "syllables"})


def make_cache_key(kind: str, syllable: int, manifest: Dict[str, Any], manifest_path: Path,
                   hash_cache: HashCache, code_files: Iterable[Path],
                   render_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Everything that must be identical for an existing output to be reused.  The
    selection for THIS syllable, the global selection settings, the model
    results, relevant code/config and the calibration/correspondence inputs are
    all hashed.  (A change to some other syllable's selection does not
    invalidate this syllable; the whole-manifest hash is still recorded in the
    sidecar for traceability.)
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "syllable": int(syllable),
        "manifest_syllable_sha256": manifest_syllable_hash(manifest, syllable),
        "manifest_global_sha256": manifest_global_hash(manifest),
        "results_sha256": hash_cache.get(RESULTS_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "vfi_sha256": hash_cache.get(VFI_PATH),
        "preprocessed_sha256": hash_cache.get(PREPROCESSED_PATH),
        "selected_source_file_identity": media_file_identity(manifest, syllable),
        "code_sha256": hash_files(code_files),
        "calibration_sha256": hash_files(CALIBRATION_FILES),
        "render_settings": render_settings,
    }


def sidecar_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(output_path.name + ".provenance.json")


def write_sidecar(output_paths: Sequence[Path], key: Dict[str, Any], manifest_path: Path,
                  hash_cache: HashCache, payload: Dict[str, Any], sidecar_target: Optional[Path] = None) -> Path:
    """Atomically write the provenance sidecar next to the (first) output."""
    output_paths = [Path(p) for p in output_paths]
    target = Path(sidecar_target) if sidecar_target else sidecar_path(output_paths[0])
    record = {
        "cache_key": key,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "outputs": {p.name: {"size": p.stat().st_size} for p in output_paths if p.exists()},
        **payload,
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, cls=NumpyEncoder))
    os.replace(tmp, target)
    return target


def cache_decision(output_paths: Sequence[Path], key: Dict[str, Any],
                   sidecar_target: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Reuse an output only if (a) all files exist with the recorded sizes, and
    (b) its sidecar cache key equals the current key.  Returns (reuse, reason);
    an existing large file alone never suppresses re-rendering.
    """
    output_paths = [Path(p) for p in output_paths]
    for p in output_paths:
        if not p.exists():
            return False, f"missing output {p.name}"
    side = Path(sidecar_target) if sidecar_target else sidecar_path(output_paths[0])
    if not side.exists():
        return False, "no provenance sidecar (legacy or unverified output)"
    try:
        rec = json.loads(side.read_text())
    except Exception as exc:
        return False, f"unreadable sidecar ({exc})"
    for p in output_paths:
        recorded = rec.get("outputs", {}).get(p.name)
        if recorded is None or recorded.get("size") != p.stat().st_size:
            return False, f"output {p.name} differs from recorded size"
    old = rec.get("cache_key", {})
    if old == key:
        return True, "cache key matches"
    changed = sorted(k for k in set(old) | set(key) if old.get(k) != key.get(k))
    return False, "cache key changed: " + ", ".join(changed)


# =============================================================================
# Installed-KPMS provenance (version + exact source hashes)
# =============================================================================
def kpms_provenance() -> Dict[str, Any]:
    """Record installed package versions and a hash of the exact KPMS source used."""
    import importlib.metadata as md
    import inspect

    import keypoint_moseq.util as kutil

    funcs = ["sample_instances", "get_instance_trajectories", "filter_centroids_headings", "filter_angle"]
    sources = {f: inspect.getsource(getattr(kutil, f)) for f in funcs}

    def ver(pkg):
        try:
            return md.version(pkg)
        except Exception:
            return None

    import numpy, scipy, sklearn
    return {
        "keypoint_moseq": ver("keypoint_moseq"),
        "jax_moseq": ver("jax_moseq"),
        "scikit_learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "util_module_path": kutil.__file__,
        "function_source_sha256": {f: hashlib.sha256(s.encode()).hexdigest() for f, s in sources.items()},
    }


# =============================================================================
# Diagnostic distances (no zero-filling of missing observations)
# =============================================================================
def rms_point_difference(a: np.ndarray, b: np.ndarray, idxs: Sequence[int]) -> float:
    """
    RMS Euclidean point error in pixels between two ``(T, K, 2)`` windows over
    keypoints ``idxs``.  Used on the SCORED (preprocessed) representation,
    where both windows are fully finite by construction.
    """
    d = np.asarray(a, dtype=np.float64)[:, idxs] - np.asarray(b, dtype=np.float64)[:, idxs]
    return float(np.sqrt(np.mean(np.sum(d * d, axis=-1))))


def confidence_aware_rms(a: np.ndarray, b: np.ndarray, valid_a: np.ndarray, valid_b: np.ndarray,
                         idxs: Sequence[int]) -> Tuple[float, int]:
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


def write_json_atomic(path: Path, obj: Any, indent: Optional[int] = 2) -> None:
    """Write JSON via a temp file + ``os.replace`` so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, cls=NumpyEncoder)
    os.replace(tmp, path)
