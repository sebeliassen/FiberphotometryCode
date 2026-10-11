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

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .clock_sync import get_session_ttl_offset, load_ttl_cache

_MAPPING_CACHE: Dict[str, np.ndarray] = {}
_CALIBRATION_DETAILS_CACHE: Dict[str, dict] = {}


def mapping_fingerprint(mapping):
    """Identify integer correspondence values together with their array layout."""
    mapping = np.asarray(mapping)
    metadata = json.dumps({"dtype": mapping.dtype.str, "shape": list(mapping.shape)},
                          sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(metadata + b"\0")
    digest.update(mapping.tobytes())
    return digest.hexdigest()


class CameraMappingContext:
    """Session lookup and in-memory mapping cache scoped to explicit inputs.

    Two contexts may reuse a session ID without sharing a calibration. Timestamp
    path/size/mtime changes invalidate computed mappings within a context.
    There are no persistent writes or filesystem discovery operations.
    """

    def __init__(self, *, ttl_cache=None, ttl_cache_path=None, offset_lookup=None,
                 max_jitter_ms=8.35):
        if sum(value is not None for value in (ttl_cache, ttl_cache_path, offset_lookup)) != 1:
            raise ValueError("Provide exactly one TTL cache, cache path, or offset lookup")
        self.ttl_cache = ttl_cache
        self.ttl_cache_path = ttl_cache_path
        self.offset_lookup = offset_lookup
        self.max_jitter_ms = float(max_jitter_ms)
        if not np.isfinite(self.max_jitter_ms) or self.max_jitter_ms < 0:
            raise ValueError("max_jitter_ms must be finite and non-negative")
        self._computed = {}

    def offset_info(self, sid):
        return get_session_ttl_offset(sid, cache=self.ttl_cache, cache_path=self.ttl_cache_path,
                                      lookup=self.offset_lookup)

    def resolve(self, sid, *, strict=False):
        info = self.offset_info(sid)
        paths = _timestamp_paths(info)
        if paths is None:
            return None, None
        records = [(path, path.stat()) for path in paths]
        key = tuple((str(path.resolve()), stat.st_size, stat.st_mtime_ns) for path, stat in records)
        key = (key, self.max_jitter_ms)
        if key not in self._computed:
            mapping, _, calibration = compute_session_camera_mapping(*paths, max_jitter_ms=self.max_jitter_ms,
                                                                      strict=strict)
            mapping.setflags(write=False)
            self._computed[key] = mapping, calibration
        mapping, calibration = self._computed[key]
        if strict and not calibration.get("is_valid_crystal", False):
            raise RuntimeError(f"Session {sid} cached mapping is not valid crystal calibration: {calibration.get('status')}")
        return mapping, dict(calibration)


def _timestamp_paths(info):
    top, bottom = (info or {}).get("top_csv"), (info or {}).get("bot_csv")
    if not top or not bottom:
        return None
    paths = Path(top), Path(bottom)
    return paths if all(path.is_file() for path in paths) else None


def _explicit_context(context, ttl_cache, ttl_cache_path, offset_info, max_jitter_ms):
    if sum(value is not None for value in (ttl_cache, ttl_cache_path, offset_info)) > 1:
        raise ValueError("Provide one TTL cache, cache path, or offset record")
    if context is not None:
        if any(value is not None for value in (ttl_cache, ttl_cache_path, offset_info)):
            raise ValueError("Provide a mapping context or offset inputs, not both")
        return context
    if offset_info is not None:
        return CameraMappingContext(offset_lookup=lambda sid: offset_info, max_jitter_ms=max_jitter_ms)
    if ttl_cache is not None or ttl_cache_path is not None:
        return CameraMappingContext(ttl_cache=ttl_cache, ttl_cache_path=ttl_cache_path,
                                    max_jitter_ms=max_jitter_ms)
    return None


def compute_session_camera_mapping(
    top_csv: Union[str, Path], bot_csv: Union[str, Path], max_jitter_ms: float = 8.35, strict: bool = False
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Computes true exposure-level correspondence using calibrated Cam_Basler_Timestamp.

    Args:
        top_csv: Path to TopCam AllTimestampsCombined CSV.
        bot_csv: Path to BotCam AllTimestampsCombined CSV.
        max_jitter_ms: Maximum permitted temporal delta (default: 8.35 ms = half frame @ 60 Hz).
        strict: If True, raises RuntimeError if crystal hardware calibration cannot be established
                (e.g., missing Basler columns, <5 valid anchors, or clock fitting failure).

    Returns:
        mapping: 1D np.ndarray of shape (N_top,) mapping TopCam frame -> BotCam frame (or -1 if dropped)
        jitter_ms: 1D np.ndarray of temporal delta in ms
        calib_details: Dictionary containing regression coefficients, anchor stats, drift metrics,
                       and explicit calibration provenance ('status' and 'is_valid_crystal').
    """
    top_csv = Path(top_csv)
    bot_csv = Path(bot_csv)

    if not top_csv.exists() or not bot_csv.exists():
        raise FileNotFoundError(f"Timestamp files not found: top={top_csv}, bot={bot_csv}")

    # Read available timestamp columns
    cols_top = pd.read_csv(top_csv, nrows=0).columns.tolist()
    cols_bot = pd.read_csv(bot_csv, nrows=0).columns.tolist()

    use_basler = ("Cam_Basler_Timestamp" in cols_top) and ("Cam_Basler_Timestamp" in cols_bot)

    if use_basler:
        df_t = pd.read_csv(top_csv, usecols=["Cam_Basler_Timestamp", "Cam_Bonsai_Timestamp"])
        df_b = pd.read_csv(bot_csv, usecols=["Cam_Basler_Timestamp", "Cam_Bonsai_Timestamp"])

        if len(df_t) == 0 or len(df_b) == 0:
            raise ValueError(f"Empty timestamp file detected: top={len(df_t)}, bot={len(df_b)}")

        t_bas = df_t["Cam_Basler_Timestamp"].to_numpy().astype(np.float64)
        b_bas = df_b["Cam_Basler_Timestamp"].to_numpy().astype(np.float64)
        t_bon = df_t["Cam_Bonsai_Timestamp"].to_numpy().astype(np.float64)
        b_bon = df_b["Cam_Bonsai_Timestamp"].to_numpy().astype(np.float64)

        # Validate finite and non-empty
        if not (
            np.all(np.isfinite(t_bas))
            and np.all(np.isfinite(b_bas))
            and np.all(np.isfinite(t_bon))
            and np.all(np.isfinite(b_bon))
        ):
            raise ValueError(f"Non-finite (NaN/Inf) timestamps found in {top_csv.name} or {bot_csv.name}")

        # Validate monotonicity of camera crystal clocks
        if np.any(np.diff(t_bas) < 0) or np.any(np.diff(b_bas) < 0):
            raise ValueError(f"Non-monotonic crystal timestamps found in {top_csv.name} or {bot_csv.name}")

        # 1. Rate-Aware Hardware Crystal Clock Calibration
        # Filter out batched / duplicate host timestamps (USB polling artifacts)
        t_valid = np.r_[True, np.diff(t_bon) > 0.005]
        b_valid = np.r_[True, np.diff(b_bon) > 0.005]

        anchors_t = []
        anchors_b = []
        margin = min(50, max(5, len(t_bon) // 10))
        span = len(t_bon) - 2 * margin
        if span > 0:
            step = max(1, min(50, span // 20))
            for i in range(margin, len(t_bon) - margin, step):
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
                slope, intercept = float(poly[0]), float(poly[1])
                ppm_drift = float((slope - 1.0) * 1e6)
                n_anchors = int(np.sum(good))
                max_res_ms = float(final_res.max())
                mean_res_ms = float(final_res.mean())
                status = "VALID_CRYSTAL"
                mode = "rate_aware_basler_crystal"
                is_valid = True
            else:
                if strict:
                    raise RuntimeError(
                        f"Strict calibration rejected: {np.sum(good)} good anchors retained (<5 required after outlier filter) for {top_csv.name}."
                    )
                slope, intercept, ppm_drift = 1.0, float(t_bas[0] - b_bas[0]), 0.0
                n_anchors, n_rej, max_res_ms, mean_res_ms = (
                    int(np.sum(good)),
                    n_rej,
                    float(res_ms.max()),
                    float(res_ms.mean()),
                )
                status = "DEGRADED_INSUFFICIENT_ANCHORS"
                mode = "insufficient_anchors_fallback"
                is_valid = False
        else:
            if strict:
                raise RuntimeError(
                    f"Strict calibration rejected: only {len(anchors_t)} anchors found (<5 required) for {top_csv.name}."
                )
            slope, intercept, ppm_drift = 1.0, float(t_bas[0] - b_bas[0]), 0.0
            n_anchors, n_rej, max_res_ms, mean_res_ms = len(anchors_t), 0, 0.0, 0.0
            status = "DEGRADED_INSUFFICIENT_ANCHORS"
            mode = "insufficient_anchors_fallback"
            is_valid = False

        calib_details = {
            "status": status,
            "mode": mode,
            "is_valid_crystal": is_valid,
            "strict_validated": strict,
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
        if strict:
            raise RuntimeError(
                f"Strict calibration rejected: missing Cam_Basler_Timestamp columns in {top_csv.name} or {bot_csv.name}."
            )
        # Fallback to Cam_Bonsai_Timestamp
        df_t = pd.read_csv(top_csv, usecols=["Cam_Bonsai_Timestamp"])
        df_b = pd.read_csv(bot_csv, usecols=["Cam_Bonsai_Timestamp"])

        if len(df_t) == 0 or len(df_b) == 0:
            raise ValueError(f"Empty timestamp file detected: top={len(df_t)}, bot={len(df_b)}")

        t_bon = df_t["Cam_Bonsai_Timestamp"].to_numpy().astype(np.float64)
        b_bon = df_b["Cam_Bonsai_Timestamp"].to_numpy().astype(np.float64)

        if not (np.all(np.isfinite(t_bon)) and np.all(np.isfinite(b_bon))):
            raise ValueError(f"Non-finite timestamps in {top_csv.name} or {bot_csv.name}")

        idx = np.searchsorted(b_bon, t_bon)
        idx = np.clip(idx, 0, len(b_bon) - 1)
        left = np.maximum(0, idx - 1)
        closer_left = np.abs(b_bon[left] - t_bon) < np.abs(b_bon[idx] - t_bon)
        best_bot = np.where(closer_left, left, idx)

        dt_ms = np.abs(b_bon[best_bot] - t_bon) * 1000.0
        valid = (dt_ms <= max_jitter_ms) & (t_bon >= b_bon[0] - 0.008) & (t_bon <= b_bon[-1] + 0.008)
        mapping = np.where(valid, best_bot, -1)
        calib_details = {
            "status": "FALLBACK_HOST_TIMESTAMPS",
            "mode": "bonsai_host_fallback",
            "is_valid_crystal": False,
            "strict_validated": strict,
            "slope": 1.0,
            "intercept": 0.0,
            "ppm_drift": 0.0,
            "n_anchors_retained": len(t_bon),
            "n_anchors_rejected": 0,
            "anchor_max_res_ms": float(dt_ms[valid].max()) if np.any(valid) else 0.0,
            "anchor_mean_res_ms": float(dt_ms[valid].mean()) if np.any(valid) else 0.0,
        }
        return mapping, dt_ms, calib_details


def get_session_mapping(sid: str, strict: bool = False, *, context=None, ttl_cache=None,
                        ttl_cache_path=None, offset_info=None, max_jitter_ms=8.35) -> Optional[np.ndarray]:
    """Map supplied inputs; omitted inputs use the historical read-only facade."""
    explicit = _explicit_context(context, ttl_cache, ttl_cache_path, offset_info, max_jitter_ms)
    if explicit is not None:
        return explicit.resolve(sid, strict=strict)[0]
    global _MAPPING_CACHE, _CALIBRATION_DETAILS_CACHE
    clean_sid = re.sub(r"_seg\d+", "", sid).replace("_tandem_23kp", "")

    if clean_sid in _MAPPING_CACHE:
        calib = _CALIBRATION_DETAILS_CACHE.get(clean_sid)
        if strict and calib and not calib.get("is_valid_crystal", False):
            raise RuntimeError(f"Session {sid} cached mapping is not valid crystal calibration: {calib.get('status')}")
        return _MAPPING_CACHE[clean_sid]

    ttl_cache = load_ttl_cache()
    info = ttl_cache.get(sid) or ttl_cache.get(clean_sid)
    if not info:
        return None

    paths = _timestamp_paths(info)
    if paths is None:
        return None

    mapping, _, calib = compute_session_camera_mapping(*paths, max_jitter_ms=max_jitter_ms, strict=strict)
    _MAPPING_CACHE[clean_sid] = mapping
    _MAPPING_CACHE[sid] = mapping
    _CALIBRATION_DETAILS_CACHE[clean_sid] = calib
    _CALIBRATION_DETAILS_CACHE[sid] = calib
    return mapping


def get_session_calibration_details(sid: str, *, context=None, ttl_cache=None, ttl_cache_path=None,
                                    offset_info=None, max_jitter_ms=8.35) -> Optional[dict]:
    """Retrieves calibration details (slope, intercept, ppm drift, residuals) for a session ID."""
    explicit = _explicit_context(context, ttl_cache, ttl_cache_path, offset_info, max_jitter_ms)
    if explicit is not None:
        return explicit.resolve(sid)[1]
    global _CALIBRATION_DETAILS_CACHE
    clean_sid = re.sub(r"_seg\d+", "", sid).replace("_tandem_23kp", "")
    if clean_sid not in _CALIBRATION_DETAILS_CACHE:
        get_session_mapping(sid)
    return _CALIBRATION_DETAILS_CACHE.get(clean_sid) or _CALIBRATION_DETAILS_CACHE.get(sid)


def top_to_bot_frame_continuous(sid: str, top_frame: int, allow_static_fallback: bool = False,
                               *, context=None, ttl_cache=None, ttl_cache_path=None,
                               offset_info=None) -> int:
    """
    Maps a TopCam raw frame index to its true synchronous BotCam frame index.

    Returns:
      - valid integer >= 0 for an exposure-matched frame
      - -1 if the frame is dropped, outside camera acquisition range, or rejected
      - -2 if session mapping / timestamp files could not be loaded
    """
    mapping = get_session_mapping(sid, context=context, ttl_cache=ttl_cache,
                                   ttl_cache_path=ttl_cache_path, offset_info=offset_info)
    if mapping is not None:
        if 0 <= top_frame < len(mapping):
            return int(mapping[top_frame])  # Explicitly preserves -1 for dropped frames!
        else:
            return -1  # Out of recording bounds

    if allow_static_fallback:
        info = (context.offset_info(sid) if context is not None else
                get_session_ttl_offset(sid, cache=ttl_cache, cache_path=ttl_cache_path, offset_info=offset_info))
        if info:
            offset = int(info.get("bot0_in_top", 0) - info.get("top0_in_bot", 0))
            return max(0, top_frame - offset)

    return -2  # Timestamp files missing / unmapped session
