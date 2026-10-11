"""
core.ttl_sync
=============
Hardware TTL & Photometry System Clock Synchronization Engine.

Solves cross-camera trigger latency and startup timing jitter across
TopCam (CenterTopCam) and BotCam (Arenas 1-4) by matching frames
against the shared master FP3002 photometry system hardware clock
(`FP3002_System_Timestamp`).

Key capabilities:
1. `get_session_ttl_offset(sid)`: Returns exact integer frame offset and startup latency.
2. `top_to_bot_frame(sid, top_frame)`: Maps any TopCam raw video frame index to its exact synchronous BotCam frame index.
3. `build_and_save_ttl_cache()`: Generates and persists `ttl_cohort_offsets.json`.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections.abc import Mapping

import numpy as np
import pandas as pd

from fpcode.adapters.multiview_compatibility import (
    ARENA_TO_BOTCAM, ARENA_TO_CHAMBER, TTL_CACHE_PATH,
    legacy_generated_ttl_path, legacy_timestamp_root,
)
from fpcode.common.cache_manifest import atomic_path

CACHE_FILE = TTL_CACHE_PATH  # historical public name, read-only compatibility resource

_OFFSET_CACHE: Optional[Dict[str, dict]] = None

def load_ttl_cache(cache_path=None, *, cache=None) -> Dict[str, dict]:
    """Read supplied offsets or a JSON file; never discover or rebuild a cache.

    Omitting both arguments selects the packaged historical cohort resource.
    A missing explicit file is an error; callers can deliberately run the builder.
    """
    if cache is not None:
        if cache_path is not None:
            raise ValueError("Supply cache or cache_path, not both")
        if not isinstance(cache, Mapping):
            raise TypeError("TTL cache must be a session-to-offset mapping")
        return cache
    global _OFFSET_CACHE
    if cache_path is None and _OFFSET_CACHE is not None:
        return _OFFSET_CACHE
    path = Path(cache_path) if cache_path is not None else CACHE_FILE
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or any(not isinstance(value, dict) for value in data.values()):
        raise ValueError(f"TTL cache must contain session-to-offset objects: {path}")
    if cache_path is None:
        _OFFSET_CACHE = data
    return data


def find_camera_timestamp_files(ts: str, arena: str, *, source_root=None,
                                arena_to_botcam=None) -> Tuple[Optional[Path], Optional[Path]]:
    """Explicit discovery operation; no normal mapping/cache read calls this."""
    root = Path(source_root) if source_root is not None else legacy_timestamp_root()
    names = ARENA_TO_BOTCAM if arena_to_botcam is None else arena_to_botcam
    botcam_name = names.get(arena)
    top_candidates = sorted(root.glob(f"**/*CenterTopCam_AllTimestampsCombined*{ts}*.csv"))
    bot_candidates = sorted(root.glob(f"**/*{botcam_name}_AllTimestampsCombined*{ts}*.csv")) if botcam_name else []

    top_p = top_candidates[0] if top_candidates else None
    bot_p = bot_candidates[0] if bot_candidates else None
    return top_p, bot_p


def compute_pair_ttl_offset(top_csv: Path, bot_csv: Path) -> dict:
    """Computes exact start offset and alignment residuals from hardware timestamps."""
    df_t = pd.read_csv(top_csv, usecols=["FP3002_System_Timestamp"])
    df_b = pd.read_csv(bot_csv, usecols=["FP3002_System_Timestamp"])
    t_fp = df_t["FP3002_System_Timestamp"].to_numpy()
    b_fp = df_b["FP3002_System_Timestamp"].to_numpy()

    dt_start_sec = float(b_fp[0] - t_fp[0])
    # Which TopCam frame matches BotCam frame 0?
    bot0_in_top = int(np.searchsorted(t_fp, b_fp[0]))
    # Which BotCam frame matches TopCam frame 0?
    top0_in_bot = int(np.searchsorted(b_fp, t_fp[0]))

    # Test residual jitter across first 100 frames
    n_test = min(100, len(b_fp), max(0, len(t_fp) - bot0_in_top))
    if n_test > 10 and bot0_in_top + n_test <= len(t_fp):
        res = (b_fp[:n_test] - t_fp[bot0_in_top : bot0_in_top + n_test]) * 1000.0
        mean_res_ms = float(np.mean(np.abs(res)))
    else:
        mean_res_ms = 0.0

    return {
        "top_total_frames": len(t_fp),
        "bot_total_frames": len(b_fp),
        "dt_start_sec": dt_start_sec,
        "bot0_in_top": bot0_in_top,
        "top0_in_bot": top0_in_bot,
        "mean_res_ms": mean_res_ms,
    }


def build_and_save_ttl_cache(out_path: Optional[Path] = None, *, timestamp_pairs=None,
                             source_root=None, arena_to_botcam=None,
                             arena_to_chamber=None) -> Dict[str, dict]:
    """Deliberately build offsets from supplied pairs, or an explicit discovery root.

    ``timestamp_pairs`` maps session IDs to ``top_csv``/``bot_csv`` metadata.
    The historical no-argument discovery operation remains available, but its
    output is under ``outputs/multiview_compatibility`` rather than package code.
    Reading that generated file requires an explicit ``cache_path``.
    """
    target_path = Path(out_path) if out_path is not None else legacy_generated_ttl_path()
    if timestamp_pairs is not None:
        if source_root is not None:
            raise ValueError("Supply timestamp_pairs or source_root, not both")
        cache_data = {}
        for sid, info in timestamp_pairs.items():
            res = compute_pair_ttl_offset(Path(info["top_csv"]), Path(info["bot_csv"]))
            cache_data[sid] = {**info, **res, "top_csv": str(info["top_csv"]), "bot_csv": str(info["bot_csv"])}
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(target_path) as temporary:
            temporary.write_text(json.dumps(cache_data, indent=2))
        return cache_data
    root = Path(source_root) if source_root is not None else legacy_timestamp_root()
    cameras = ARENA_TO_BOTCAM if arena_to_botcam is None else arena_to_botcam
    chambers = ARENA_TO_CHAMBER if arena_to_chamber is None else arena_to_chamber
    all_ts_files = sorted(root.glob("**/*AllTimestampsCombined*.csv"))

    ts_map: Dict[str, dict] = {}
    for p in all_ts_files:
        m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})", p.name)
        if not m:
            continue
        ts = m.group(1)
        d = ts_map.setdefault(ts, {})
        if "CenterTopCam" in p.name:
            d["top"] = p
        for arena, camera in cameras.items():
            if camera in p.name:
                d[arena] = p

    cache_data: Dict[str, dict] = {}
    for ts, d in ts_map.items():
        if "top" not in d:
            continue
        top_csv = d["top"]
        for arena_name, chamber_name in chambers.items():
            if arena_name in d:
                bot_csv = d[arena_name]
                try:
                    res = compute_pair_ttl_offset(top_csv, bot_csv)
                    sid = f"{ts}_{chamber_name}"
                    res["ts"] = ts
                    res["arena"] = arena_name
                    res["chamber"] = chamber_name
                    res["top_csv"] = str(top_csv)
                    res["bot_csv"] = str(bot_csv)
                    cache_data[sid] = res
                    cache_data[f"{sid}_tandem_23kp"] = res
                except Exception as e:
                    print(f"[WARN] Error computing TTL offset for {ts} {arena_name}: {e}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(target_path) as temporary:
        temporary.write_text(json.dumps(cache_data, indent=2))
    print(f"✅ Saved TTL offset cache to {target_path} ({len(cache_data)} keys)")
    return cache_data


def _validate_offset_info(info, sid):
    if not isinstance(info, Mapping) or not info or not any(
        key in info for key in ("top_csv", "bot_csv", "bot0_in_top", "top0_in_bot")
    ):
        raise ValueError(f"Malformed TTL offset record for session {sid}")
    return info


def get_session_ttl_offset(sid: str, *, cache=None, cache_path=None, offset_info=None, lookup=None) -> dict:
    """Resolve one supplied source; absent explicit sessions raise ``KeyError``.

    The historical call without source arguments retains its missing-session
    zero-offset policy. Generic inputs never silently inherit that policy.
    """
    source_count = sum(value is not None for value in (cache, cache_path, offset_info, lookup))
    if source_count > 1:
        raise ValueError("Provide one TTL cache, cache path, offset record, or lookup")
    if offset_info is not None:
        return _validate_offset_info(offset_info, sid)
    if lookup is not None:
        return _validate_offset_info(lookup(sid), sid)
    cache = load_ttl_cache(cache_path, cache=cache)
    clean_sid = re.sub(r"_seg\d+", "", sid).replace("_tandem_23kp", "")
    if sid in cache:
        return _validate_offset_info(cache[sid], sid) if source_count else cache[sid]
    if clean_sid in cache:
        return _validate_offset_info(cache[clean_sid], sid) if source_count else cache[clean_sid]
    if source_count:
        raise KeyError(f"Session {sid} is absent from the supplied TTL cache")
    return {"bot0_in_top": 0, "top0_in_bot": 0, "dt_start_sec": 0.0, "mean_res_ms": 0.0}


def top_to_bot_frame(sid: str, top_frame: int, **offset_options) -> int:
    """
    Maps a TopCam raw video frame index to the synchronous BotCam raw video frame index.

    Formula:
        offset = bot0_in_top - top0_in_bot
        BotCam Frame = TopCam Frame - offset.
    """
    offset_info = get_session_ttl_offset(sid, **offset_options)
    offset = offset_info.get("bot0_in_top", 0) - offset_info.get("top0_in_bot", 0)
    bot_frame = top_frame - offset
    return max(0, bot_frame)


def bot_to_top_frame(sid: str, bot_frame: int, **offset_options) -> int:
    """Maps a BotCam raw video frame index to the synchronous TopCam raw video frame index."""
    offset_info = get_session_ttl_offset(sid, **offset_options)
    offset = offset_info.get("bot0_in_top", 0) - offset_info.get("top0_in_bot", 0)
    return bot_frame + offset


if __name__ == "__main__":
    print("Building and caching cohort TTL offsets...")
    build_and_save_ttl_cache()
