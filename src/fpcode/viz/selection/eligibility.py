"""Visualization quality rules for explicitly described landmark groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from fpcode.tracking.quality import supported_track_mask
from fpcode.viz.context import LandmarkGroups
from fpcode.viz.selection.bouts import frames_from_seconds


def movie_context_ok(start: int, n_frames: int, pre: int, post: int) -> bool:
    """True iff the complete half-open context [start-pre, start+post) exists."""
    return int(start) >= int(pre) and int(start) + int(post) <= int(n_frames)


def require_fused_qc(fused_data, base_sid: str, skip: bool = False):
    """Missing tracking evidence is an error, never an implicit QC pass."""
    if fused_data is None and not skip:
        raise RuntimeError(f"Original fused tracks unavailable for {base_sid}; QC cannot be certified")
    return fused_data


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
    return supported_track_mask(xy, conf, threshold=conf_threshold)


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


def summarize_window_qc(valid_window: np.ndarray, *, groups: LandmarkGroups) -> Dict[str, Any]:
    """
    Per-group QC summary of a ``(W, 21)`` supported-point mask.  Reports
    per-paw valid fractions / maximum unsupported gaps so any rejection can be
    traced to a specific landmark.
    """
    frac, gap, W = window_frac_gap(valid_window)
    return summarize_frac_gap(frac, gap, W, groups=groups)


def summarize_frac_gap(frac: np.ndarray, gap: np.ndarray, W: int, *, groups: LandmarkGroups) -> Dict[str, Any]:
    """Same summary as ``summarize_window_qc`` from precomputed (fraction, gap) vectors."""
    frac = np.asarray(frac, dtype=np.float64)
    gap = np.asarray(gap)
    if frac.ndim != 1 or gap.shape != frac.shape:
        raise ValueError("QC fraction and gap vectors must have matching one-dimensional shapes")
    dorsal_frac, dorsal_gap = frac[list(groups.dorsal)], gap[list(groups.dorsal)]
    paw_frac, paw_gap = frac[list(groups.paws)], gap[list(groups.paws)]
    front_frac, front_gap = frac[list(groups.front_paws)], gap[list(groups.front_paws)]
    hind_frac, hind_gap = frac[list(groups.hind_paws)], gap[list(groups.hind_paws)]
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
        "per_paw_valid_fraction": {n: float(f) for n, f in zip(groups.paw_names, paw_frac)},
        "per_paw_max_gap_frames": {n: int(g) for n, g in zip(groups.paw_names, paw_gap)},
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
