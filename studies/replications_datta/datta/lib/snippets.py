"""
Snippets: Logic for Event-Triggered Analysis (Panel 1f).
- Event Detection (Syllable Onset, Movement Initiation)
- Window Extraction
- Signal Processing (Smoothing, Derivs)
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from fpcode.behavior.kinematics import detect_movement_initiations as _movement_initiations


def detect_syllable_onsets(df: pd.DataFrame, syllable_col: str = "labels") -> np.ndarray:
    """
    Returns DataFrame integer indices where the syllable changes to a new value.
    Ignores NaN transitions.
    """
    if syllable_col not in df.columns:
        return np.array([], dtype=int)

    syl = df[syllable_col]
    # Check for change AND not na
    # Use fillna(False) to handle pd.NA ambiguity
    is_change = syl.ne(syl.shift(1))
    is_valid = syl.notna()

    change = (is_change & is_valid).fillna(False).astype(bool)

    # We might want to filter out 'usage' or silence syllables if needed?
    # Legacy just takes all changes.
    return np.where(change.to_numpy())[0]

def detect_movement_initiations(df: pd.DataFrame, fs: float, vel_col: str = "velocity_px_s") -> np.ndarray:
    """
    Markowitz 2022 Methods for Movement Initiation:
      1. Z-score 2D velocity (global)
      2. Smooth with 1.67s boxcar
      3. Take derivative (acceleration-ish)
      4. Cube it (Changepoint score)
      5. Find peaks (height=1, width=1, prominence=1)
    """
    if vel_col not in df.columns:
        return np.array([], dtype=int)

    return _movement_initiations(df[vel_col].to_numpy(dtype=float), fs,
                                derivative_scale="per_sample", missing="propagate")

def extract_windows(df: pd.DataFrame,
                   onsets: np.ndarray,
                   window_sec: Tuple[float, float],
                   fs: float,
                   include_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extracts time-aligned windows around onset indices.
    Returns a long-form DataFrame with 'rel_time' and 'snippet_id'.
    """
    if len(onsets) == 0:
        return pd.DataFrame()

    n_samples = len(df)
    pre_samples = int(window_sec[0] * fs)
    post_samples = int(window_sec[1] * fs)

    snippets = []

    # Convert scalar args to samples once
    window_len = post_samples - pre_samples + 1
    t_vec = np.linspace(window_sec[0], window_sec[1], window_len)

    if include_cols:
        sub_df = df[include_cols]
    else:
        sub_df = df

    for i, idx in enumerate(onsets):
        start = idx + pre_samples
        stop = idx + post_samples

        # Bounds check
        if start < 0 or stop >= n_samples:
            continue

        # Slicing is exclusive at end for iloc usually, but let's be precise
        # We want [start, stop] inclusive?
        # t_vec has length (post - pre + 1)
        # So slice should be [start : stop+1]

        chunk = sub_df.iloc[start : stop+1].copy()

        if len(chunk) != window_len:
            continue

        chunk["time"] = t_vec
        chunk["snippet_id"] = i
        snippets.append(chunk)

    if not snippets:
        return pd.DataFrame()

    return pd.concat(snippets, ignore_index=True)
