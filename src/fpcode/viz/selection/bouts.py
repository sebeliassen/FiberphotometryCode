"""Visualization selection / bouts; historical V2 behavior preserved."""

from __future__ import annotations

import re
from typing import Any, Dict, Sequence, Tuple

import numpy as np


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
