# utils_timestamps.py
from __future__ import annotations  # optional but safe; keeps annotations from being evaluated
from typing import Optional, Union, Tuple, Sequence  # add this
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # so imports don't configure logging


# known candidates in AllTimestampsCombined
CANDIDATE_TS_COLS = [
    "Cam_Basler_Timestamp",
    "Cam_Bonsai_Timestamp",
    "FP3002_System_Timestamp",
    "FP3002_Bonsai_Timestamp",
]

def _infer_seconds_scale(vals: np.ndarray) -> float:
    """Return divisor to convert vals to seconds."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 1.0
    med = float(np.median(np.abs(v)))
    # ns-sized
    if med > 1e10:  # ~1e12 typical
        return 1e9
    # µs
    if 1e7 < med <= 1e10:
        return 1e6
    # ms
    if 1e4 < med <= 1e7:
        return 1e3
    # Excel/Matlab day count (~4.4e4)
    if 3e4 < med < 8e4:
        return 1/86400.0  # special-case multiply
    return 1.0

def _effective_hz(vals_s: np.ndarray) -> float | None:
    v = np.asarray(vals_s, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return None
    dt = np.diff(v)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        return None
    med = float(np.median(dt))
    if med <= 0:
        return None
    return 1.0 / med

def load_per_frame_cam_timestamps_seconds(
    all_ts_csv: str,
    target_fps: float = 60.0,
    tolerance: float = 0.15  # ±15%
) -> np.ndarray | None:
    """
    Load AllTimestampsCombined and return a seconds array that’s ~one value per behavior frame.
    Picks the timestamp column whose effective Hz is closest to target_fps within tolerance.
    """
    try:
        df = pd.read_csv(all_ts_csv)
    except Exception as e:
        logger.error(f"Failed to read {all_ts_csv}: {e}")
        return None

    best = None
    best_err = None
    for col in CANDIDATE_TS_COLS:
        if col not in df.columns:
            continue
        raw = pd.to_numeric(df[col], errors="coerce").to_numpy()
        raw = raw[np.isfinite(raw)]
        if raw.size < 3:
            continue

        scale = _infer_seconds_scale(raw)
        if scale == 1/86400.0:
            vals_s = raw * 86400.0
        else:
            vals_s = raw / scale

        hz = _effective_hz(vals_s)
        if hz is None:
            continue

        err = abs(hz - target_fps) / target_fps
        logger.debug(f"{col}: inferred {hz:.2f} Hz (err {err*100:.1f}%)")
        if err <= tolerance and (best is None or err < best_err):
            best = vals_s
            best_err = err

    if best is None:
        logger.warning(
            f"No timestamp column near ~{target_fps} Hz found in {all_ts_csv}. "
            f"Available: {', '.join([c for c in CANDIDATE_TS_COLS if c in df.columns])}. "
            f"Falling back to None (will use linear index mapping)."
        )
        return None

    return best
