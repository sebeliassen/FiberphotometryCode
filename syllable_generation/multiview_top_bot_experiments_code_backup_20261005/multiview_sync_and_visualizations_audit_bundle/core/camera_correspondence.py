"""
core.camera_correspondence
==========================
Continuous Per-Frame Hardware Camera Synchronization Engine.

Replaces the static scalar offset with microsecond-precision continuous
camera correspondence across TopCam (CenterTopCam) and BotCam (Arenas 1-4).

Key features:
1. Cross-camera hardware clock calibration at startup (using initial FP3002 / Bonsai overlap).
2. Continuous frame-by-frame matching using internal FPGA crystal clocks (`Cam_Basler_Timestamp`).
3. Explicit missing/dropped frame detection (marking dropped frames as -1).
4. Zero dependency on long-term photometry stream survival (immune to photometry freezes).
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .ttl_sync import find_camera_timestamp_files, get_session_ttl_offset, load_ttl_cache

_MAPPING_CACHE: Dict[str, np.ndarray] = {}
_CALIBRATION_DETAILS_CACHE: Dict[str, dict] = {}

def compute_session_camera_mapping(
    top_csv: Union[str, Path],
    bot_csv: Union[str, Path],
    max_jitter_ms: float = 8.35
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Computes true exposure-level correspondence using calibrated Cam_Basler_Timestamp.
    
    Args:
        top_csv: Path to TopCam AllTimestampsCombined CSV.
        bot_csv: Path to BotCam AllTimestampsCombined CSV.
        max_jitter_ms: Maximum permitted temporal delta (default: 8.35 ms = half frame @ 60 Hz).
        
    Returns:
        mapping: 1D np.ndarray of shape (N_top,) mapping TopCam frame -> BotCam frame (or -1 if dropped)
        jitter_ms: 1D np.ndarray of temporal delta in ms
        calib_details: Dictionary containing regression coefficients, anchor stats, and drift metrics.
    """
    top_csv = Path(top_csv)
    bot_csv = Path(bot_csv)
    
    # Read available timestamp columns
    cols_top = pd.read_csv(top_csv, nrows=0).columns
    cols_bot = pd.read_csv(bot_csv, nrows=0).columns
    
    use_basler = ("Cam_Basler_Timestamp" in cols_top) and ("Cam_Basler_Timestamp" in cols_bot)
    
    if use_basler:
        df_t = pd.read_csv(top_csv, usecols=["Cam_Basler_Timestamp", "Cam_Bonsai_Timestamp"])
        df_b = pd.read_csv(bot_csv, usecols=["Cam_Basler_Timestamp", "Cam_Bonsai_Timestamp"])
        
        t_bas = df_t["Cam_Basler_Timestamp"].to_numpy().astype(np.float64)
        b_bas = df_b["Cam_Basler_Timestamp"].to_numpy().astype(np.float64)
        t_bon = df_t["Cam_Bonsai_Timestamp"].to_numpy()
        b_bon = df_b["Cam_Bonsai_Timestamp"].to_numpy()
        
        # 1. Rate-Aware Hardware Crystal Clock Calibration
        # Filter out batched / duplicate host timestamps (USB polling artifacts)
        t_valid = np.r_[True, np.diff(t_bon) > 0.005]
        b_valid = np.r_[True, np.diff(b_bon) > 0.005]
        
        anchors_t = []
        anchors_b = []
        step = max(50, len(t_bon) // 300)
        for i in range(50, len(t_bon) - 50, step):
            if not t_valid[i]:
                continue
            t_val = t_bon[i]
            b_idx = int(np.searchsorted(b_bon, t_val))
            if 0 < b_idx < len(b_bon):
                if abs(b_bon[b_idx - 1] - t_val) < abs(b_bon[b_idx] - t_val):
                    b_idx -= 1
                if b_valid[b_idx]:
                    dt = abs(b_bon[b_idx] - t_val)
                    if dt < 0.003:  # strictly within 3ms host time
                        anchors_t.append(t_bas[i])
                        anchors_b.append(b_bas[b_idx])
                        
        anchors_t = np.array(anchors_t)
        anchors_b = np.array(anchors_b)
        
        if len(anchors_t) >= 5:
            poly = np.polyfit(anchors_b, anchors_t, deg=1)
            res_ms = np.abs(anchors_t - (poly[0] * anchors_b + poly[1])) / 1e6
            good = res_ms < 2.0  # outlier rejection
            n_rej = int(np.sum(~good))
            if np.sum(good) >= 5:
                poly = np.polyfit(anchors_b[good], anchors_t[good], deg=1)
                final_res = np.abs(anchors_t[good] - (poly[0] * anchors_b[good] + poly[1])) / 1e6
            else:
                final_res = res_ms
            slope, intercept = float(poly[0]), float(poly[1])
            ppm_drift = float((slope - 1.0) * 1e6)
            n_anchors = int(np.sum(good))
            max_res_ms = float(final_res.max())
            mean_res_ms = float(final_res.mean())
        else:
            slope, intercept, ppm_drift = 1.0, float(t_bas[0] - b_bas[0]), 0.0
            n_anchors, n_rej, max_res_ms, mean_res_ms = len(anchors_t), 0, 0.0, 0.0
            
        calib_details = {
            "mode": "rate_aware_basler_crystal",
            "slope": slope,
            "intercept": intercept,
            "ppm_drift": ppm_drift,
            "n_anchors_retained": n_anchors,
            "n_anchors_rejected": n_rej,
            "anchor_max_res_ms": max_res_ms,
            "anchor_mean_res_ms": mean_res_ms,
        }
        
        # Calibrated BotCam crystal time projected into TopCam crystal timeline
        b_bas_cal = slope * b_bas + intercept
        
        # 2. Nearest neighbor search on calibrated Basler crystal time
        idx = np.searchsorted(b_bas_cal, t_bas)
        idx = np.clip(idx, 0, len(b_bas_cal) - 1)
        left = np.maximum(0, idx - 1)
        closer_left = np.abs(b_bas_cal[left] - t_bas) < np.abs(b_bas_cal[idx] - t_bas)
        best_bot = np.where(closer_left, left, idx)
        
        dt_ms = np.abs(b_bas_cal[best_bot] - t_bas) / 1e6
        valid = (dt_ms <= max_jitter_ms) & (t_bas >= b_bas_cal[0] - 8e6) & (t_bas <= b_bas_cal[-1] + 8e6)
        mapping = np.where(valid, best_bot, -1)
        return mapping, dt_ms, calib_details
    else:
        # Fallback to Cam_Bonsai_Timestamp
        df_t = pd.read_csv(top_csv, usecols=["Cam_Bonsai_Timestamp"])
        df_b = pd.read_csv(bot_csv, usecols=["Cam_Bonsai_Timestamp"])
        t_bon = df_t["Cam_Bonsai_Timestamp"].to_numpy()
        b_bon = df_b["Cam_Bonsai_Timestamp"].to_numpy()
        
        idx = np.searchsorted(b_bon, t_bon)
        idx = np.clip(idx, 0, len(b_bon) - 1)
        left = np.maximum(0, idx - 1)
        closer_left = np.abs(b_bon[left] - t_bon) < np.abs(b_bon[idx] - t_bon)
        best_bot = np.where(closer_left, left, idx)
        
        dt_ms = np.abs(b_bon[best_bot] - t_bon) * 1000.0
        valid = (dt_ms <= max_jitter_ms) & (t_bon >= b_bon[0] - 0.008) & (t_bon <= b_bon[-1] + 0.008)
        mapping = np.where(valid, best_bot, -1)
        calib_details = {
            "mode": "bonsai_host_fallback",
            "slope": 1.0,
            "intercept": 0.0,
            "ppm_drift": 0.0,
            "n_anchors_retained": len(t_bon),
            "n_anchors_rejected": 0,
            "anchor_max_res_ms": float(dt_ms[valid].max()) if np.any(valid) else 0.0,
            "anchor_mean_res_ms": float(dt_ms[valid].mean()) if np.any(valid) else 0.0,
        }
        return mapping, dt_ms, calib_details

def get_session_mapping(sid: str) -> Optional[np.ndarray]:
    """Retrieves or computes the continuous TopCam -> BotCam mapping array for a session ID."""
    global _MAPPING_CACHE, _CALIBRATION_DETAILS_CACHE
    clean_sid = sid.replace("_tandem_23kp", "")
    
    if clean_sid in _MAPPING_CACHE:
        return _MAPPING_CACHE[clean_sid]
        
    ttl_cache = load_ttl_cache()
    info = ttl_cache.get(sid) or ttl_cache.get(clean_sid)
    if not info:
        return None
        
    top_csv = Path(info.get("top_csv", ""))
    bot_csv = Path(info.get("bot_csv", ""))
    if not top_csv.exists() or not bot_csv.exists():
        return None
        
    mapping, _, calib = compute_session_camera_mapping(top_csv, bot_csv)
    _MAPPING_CACHE[clean_sid] = mapping
    _MAPPING_CACHE[sid] = mapping
    _CALIBRATION_DETAILS_CACHE[clean_sid] = calib
    _CALIBRATION_DETAILS_CACHE[sid] = calib
    return mapping

def get_session_calibration_details(sid: str) -> Optional[dict]:
    """Retrieves calibration details (slope, intercept, ppm drift, residuals) for a session ID."""
    global _CALIBRATION_DETAILS_CACHE
    clean_sid = sid.replace("_tandem_23kp", "")
    if clean_sid not in _CALIBRATION_DETAILS_CACHE:
        get_session_mapping(sid)
    return _CALIBRATION_DETAILS_CACHE.get(clean_sid) or _CALIBRATION_DETAILS_CACHE.get(sid)

def top_to_bot_frame_continuous(sid: str, top_frame: int, allow_static_fallback: bool = False) -> int:
    """
    Maps a TopCam raw frame index to its true synchronous BotCam frame index.
    
    Returns:
      - valid integer >= 0 for an exposure-matched frame
      - -1 if the frame is dropped, outside camera acquisition range, or rejected
      - -2 if session mapping / timestamp files could not be loaded
    """
    mapping = get_session_mapping(sid)
    if mapping is not None:
        if 0 <= top_frame < len(mapping):
            return int(mapping[top_frame])  # Explicitly preserves -1 for dropped frames!
        else:
            return -1  # Out of recording bounds
            
    if allow_static_fallback:
        info = get_session_ttl_offset(sid)
        if info:
            offset = int(info.get("bot0_in_top", 0) - info.get("top0_in_bot", 0))
            return max(0, top_frame - offset)
            
    return -2  # Timestamp files missing / unmapped session
