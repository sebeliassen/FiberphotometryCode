"""
fpcode.photometry.processing
============================
Signal grouping, confidence interval estimation, and multi-stream alignment
for photometry, Bonsai behavioral events, and camera exposure frames.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def group_signals(
    sessions: List[Any], key_fn: Callable[[Any, str], Tuple[Any, ...]]
) -> Dict[Tuple[Any, ...], np.ndarray]:
    """
    Collects all non-empty signal_matrix arrays across sessions into groups.
    key_fn: (session, signal_key) -> grouping tuple (e.g. (genotype, event, region))
    Returns dict: grouping_key -> np.vstack(matrices)
    """
    groups = defaultdict(list)
    for s in sessions:
        if not hasattr(s, "signal_info") or not s.signal_info:
            continue
        for sig_key, info in s.signal_info.items():
            mat = info.get("signal_matrix")
            if mat is not None and getattr(mat, "size", 0) > 0:
                groups[key_fn(s, sig_key)].append(mat)
    return {k: np.vstack(v) for k, v in groups.items() if v}


def compute_mean_ci(mat: np.ndarray, alpha: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given mat shape (n_trials, n_time), returns (mean, lower_ci, upper_ci).
    """
    mat = np.asarray(mat)
    if mat.ndim != 2 or mat.shape[0] == 0:
        empty = np.zeros(mat.shape[-1] if mat.ndim > 0 else 0)
        return empty, empty, empty

    ys = np.mean(mat, axis=0)
    sem = np.std(mat, axis=0) / np.sqrt(mat.shape[0])
    lo, hi = stats.norm.interval(alpha, loc=ys, scale=sem)
    return ys, lo, hi


def merge_phot_bonsai_cam(
    phot_df: pd.DataFrame,
    bonsai_df: pd.DataFrame,
    freq_label: str,
    cam_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aligns raw photometry fluorometer samples with Bonsai tracking timestamps
    and Basler camera exposure frame indices using nearest-neighbor timestamp matching.
    """
    phot = phot_df.copy()
    bonsai_col = f"TimestampBonsai_{freq_label}"
    phot[bonsai_col] = bonsai_df[bonsai_col]

    merged = pd.merge_asof(
        left=phot,
        right=cam_df[["cam_frame_num", "BonsaiTrackingTimestamp"]],
        left_on=bonsai_col,
        right_on="BonsaiTrackingTimestamp",
        direction="nearest",
    )
    return merged.drop(columns=[bonsai_col])
