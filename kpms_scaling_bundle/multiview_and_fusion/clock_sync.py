#!/usr/bin/env python3
"""
clock_sync.py
-------------
Hardware Clock Synchronization Engine for Multi-View TopCam + BotCam & Photometry.

Provides:
1. align_camera_timestamps(top_ts, bot_ts, max_jitter_sec=0.025):
   Vectorized nearest-neighbor alignment using FP3002_System_Timestamp.
2. align_tracking_dataframes(df_top, df_bot, top_ts=None, bot_ts=None):
   Aligns BotCam tracking DataFrame rows to TopCam frame indices, safely
   inserting NaNs for any dropped frames to prevent temporal phase shear.
3. load_timestamps_from_csv(allts_csv_path):
   Loads FP3002_System_Timestamp from AllTimestampsCombined CSVs.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

def load_timestamps_from_csv(allts_csv_path: str | Path) -> np.ndarray:
    """
    Extracts FP3002_System_Timestamp from an AllTimestampsCombined CSV.
    Falls back to Cam_Bonsai_Timestamp if needed.
    """
    p = Path(allts_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"AllTimestampsCombined CSV not found: {p}")

    header = pd.read_csv(p, nrows=0).columns
    target_col = None
    for cand in ["FP3002_System_Timestamp", "SystemTimestamp", "Cam_Bonsai_Timestamp", "BonsaiTimestamp"]:
        if cand in header:
            target_col = cand
            break

    if target_col is None:
        raise KeyError(f"No recognized timestamp column found in {p.name}. Available: {list(header)}")

    try:
        df = pd.read_csv(p, usecols=[target_col], engine="pyarrow")
    except Exception:
        df = pd.read_csv(p, usecols=[target_col])

    ts = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    return ts

def align_camera_timestamps(
    top_ts: np.ndarray,
    bot_ts: np.ndarray,
    max_jitter_sec: float = 0.025
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Aligns BotCam frame indices to TopCam frame indices using nearest-neighbor search.

    Args:
        top_ts: 1D array of TopCam timestamps (seconds).
        bot_ts: 1D array of BotCam timestamps (seconds).
        max_jitter_sec: Maximum permitted delta between matched frames (default: 25ms, i.e. 2 frames @ 80Hz).

    Returns:
        bot_indices_mapped: 1D integer array of shape (len(top_ts),) where element i
                            is the BotCam frame index matching TopCam frame i, or -1 if dropped.
        stats: Diagnostic summary dict.
    """
    top_ts = np.asarray(top_ts, dtype=np.float64).ravel()
    bot_ts = np.asarray(bot_ts, dtype=np.float64).ravel()

    n_top = len(top_ts)
    n_bot = len(bot_ts)

    if n_top == 0 or n_bot == 0:
        raise ValueError(f"Empty timestamp array: n_top={n_top}, n_bot={n_bot}")

    # Binary search for closest BotCam frame for each TopCam frame
    idx = np.searchsorted(bot_ts, top_ts, side="left")
    idx0 = np.clip(idx - 1, 0, n_bot - 1)
    idx1 = np.clip(idx,     0, n_bot - 1)

    left_dist = np.abs(top_ts - bot_ts[idx0])
    right_dist = np.abs(bot_ts[idx1] - top_ts)

    best_idx = np.where(left_dist <= right_dist, idx0, idx1)
    min_dist = np.where(left_dist <= right_dist, left_dist, right_dist)

    # Valid matches within max jitter
    valid_mask = min_dist <= max_jitter_sec
    bot_indices_mapped = np.where(valid_mask, best_idx, -1)

    stats = {
        "n_top_frames": int(n_top),
        "n_bot_frames": int(n_bot),
        "matched_frames": int(valid_mask.sum()),
        "dropped_bot_frames": int((~valid_mask).sum()),
        "max_jitter_observed_ms": float(np.max(min_dist[valid_mask]) * 1000.0) if valid_mask.any() else 0.0,
        "mean_jitter_observed_ms": float(np.mean(min_dist[valid_mask]) * 1000.0) if valid_mask.any() else 0.0,
    }

    return bot_indices_mapped, stats

def align_tracking_dataframes(
    df_top: pd.DataFrame,
    df_bot: pd.DataFrame,
    top_ts: Optional[np.ndarray] = None,
    bot_ts: Optional[np.ndarray] = None,
    max_jitter_sec: float = 0.025
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Time-locks BotCam tracking rows to TopCam tracking rows.
    If timestamps are provided, performs clock-hardened alignment.
    If timestamps are None, aligns via 1:1 index truncation/padding.

    Returns:
        df_top_aligned: pd.DataFrame of length N_top
        df_bot_aligned: pd.DataFrame of length N_top (with NaNs at dropped indices)
        stats: Diagnostic dictionary
    """
    n_top = len(df_top)
    n_bot = len(df_bot)

    if top_ts is not None and bot_ts is not None:
        bot_idx_map, stats = align_camera_timestamps(top_ts, bot_ts, max_jitter_sec=max_jitter_sec)
        
        # Build aligned BotCam dataframe
        # Where map == -1, rows will be NaN
        valid_mask = bot_idx_map >= 0
        valid_indices = bot_idx_map[valid_mask]
        
        # Create empty DataFrame with same columns and index 0..N_top-1
        df_bot_aligned = pd.DataFrame(np.nan, index=np.arange(n_top), columns=df_bot.columns)
        if len(valid_indices) > 0:
            df_bot_aligned.iloc[valid_mask] = df_bot.iloc[valid_indices].values
            
        df_top_aligned = df_top.copy().reset_index(drop=True)
    else:
        # 1:1 mapping fallback (e.g. when AllTimestampsCombined is not available)
        common_len = min(n_top, n_bot)
        df_top_aligned = df_top.iloc[:common_len].copy().reset_index(drop=True)
        df_bot_aligned = df_bot.iloc[:common_len].copy().reset_index(drop=True)
        stats = {
            "n_top_frames": n_top,
            "n_bot_frames": n_bot,
            "matched_frames": common_len,
            "dropped_bot_frames": abs(n_top - n_bot),
            "alignment_mode": "1:1_fallback"
        }

    return df_top_aligned, df_bot_aligned, stats

if __name__ == "__main__":
    # Self-test on mock clock data
    print("Testing clock_sync module...")
    top_clock = np.linspace(100.0, 200.0, 8000)  # 8000 frames @ 80Hz (dt = 0.0125s)
    # Simulate BotCam with a dropped frame at frame 500
    bot_clock = np.delete(top_clock + 0.0005, 500)  # 0.5ms jitter, missing frame 500

    mapped, stats = align_camera_timestamps(top_clock, bot_clock, max_jitter_sec=0.015)
    print("Self-test stats:", stats)
    assert mapped[499] == 499
    # Frame 500 on TopCam should map to either -1 or frame 500 on BotCam (which was original 501)
    print(f"Top frame 499 -> Bot frame {mapped[499]}")
    print(f"Top frame 500 -> Bot frame {mapped[500]}")
    print(f"Top frame 501 -> Bot frame {mapped[501]}")
    print("All clock_sync self-tests passed successfully.")
