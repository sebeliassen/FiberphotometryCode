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

import numpy as np
import pandas as pd

try:
    from ..common.paths import resolve_data_path
except ImportError:

    def resolve_data_path(p: str) -> Path:
        return Path(p)


HDD_DIR = (
    Path("/mnt/hdd/karen_project") if Path("/mnt/hdd/karen_project").exists() else resolve_data_path("karen_project")
)
CACHE_FILE = Path(__file__).resolve().parent / "ttl_cohort_offsets.json"

_OFFSET_CACHE: Optional[Dict[str, dict]] = None

ARENA_TO_BOTCAM = {
    "Arena1": "BotCam1",
    "Arena2": "BotCam2",
    "Arena3": "BotCam3",
    "Arena4": "BotCam4",
}


def load_ttl_cache() -> Dict[str, dict]:
    """Loads the cached TTL offsets dictionary."""
    global _OFFSET_CACHE
    if _OFFSET_CACHE is not None:
        return _OFFSET_CACHE
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            _OFFSET_CACHE = json.load(f)
            return _OFFSET_CACHE
    # If not built yet, build it
    _OFFSET_CACHE = build_and_save_ttl_cache()
    return _OFFSET_CACHE


def find_camera_timestamp_files(ts: str, arena: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Locates the TopCam and BotCam timestamp CSV files for a given timestamp and arena."""
    botcam_name = ARENA_TO_BOTCAM.get(arena)
    top_candidates = list(HDD_DIR.glob(f"**/*CenterTopCam_AllTimestampsCombined*{ts}*.csv"))
    bot_candidates = list(HDD_DIR.glob(f"**/*{botcam_name}_AllTimestampsCombined*{ts}*.csv")) if botcam_name else []

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


def build_and_save_ttl_cache(out_path: Optional[Path] = None) -> Dict[str, dict]:
    """Scans all cohort timestamps and builds the master TTL offset cache."""
    target_path = out_path or CACHE_FILE
    all_ts_files = list(HDD_DIR.glob("**/*AllTimestampsCombined*.csv"))

    ts_map: Dict[str, dict] = {}
    for p in all_ts_files:
        m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})", p.name)
        if not m:
            continue
        ts = m.group(1)
        d = ts_map.setdefault(ts, {})
        if "CenterTopCam" in p.name:
            d["top"] = p
        for a_i in range(1, 5):
            if f"BotCam{a_i}" in p.name:
                d[f"Arena{a_i}"] = p

    cache_data: Dict[str, dict] = {}
    for ts, d in ts_map.items():
        if "top" not in d:
            continue
        top_csv = d["top"]
        for arena_name, chamber_name in [("Arena1", "top_left"), ("Arena2", "bottom_left"), ("Arena3", "bottom_right")]:
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

    with open(target_path, "w") as f:
        json.dump(cache_data, f, indent=2)
    print(f"✅ Saved TTL offset cache to {target_path} ({len(cache_data)} keys)")
    return cache_data


def get_session_ttl_offset(sid: str) -> dict:
    """Retrieves the TTL offset metadata for a session ID."""
    cache = load_ttl_cache()
    clean_sid = sid.replace("_tandem_23kp", "")
    if sid in cache:
        return cache[sid]
    if clean_sid in cache:
        return cache[clean_sid]
    return {"bot0_in_top": 0, "top0_in_bot": 0, "dt_start_sec": 0.0, "mean_res_ms": 0.0}


def top_to_bot_frame(sid: str, top_frame: int) -> int:
    """
    Maps a TopCam raw video frame index to the synchronous BotCam raw video frame index.

    Formula:
        offset = bot0_in_top - top0_in_bot
        BotCam Frame = TopCam Frame - offset.
    """
    offset_info = get_session_ttl_offset(sid)
    offset = offset_info.get("bot0_in_top", 0) - offset_info.get("top0_in_bot", 0)
    bot_frame = top_frame - offset
    return max(0, bot_frame)


def bot_to_top_frame(sid: str, bot_frame: int) -> int:
    """Maps a BotCam raw video frame index to the synchronous TopCam raw video frame index."""
    offset_info = get_session_ttl_offset(sid)
    offset = offset_info.get("bot0_in_top", 0) - offset_info.get("top0_in_bot", 0)
    return bot_frame + offset


if __name__ == "__main__":
    print("Building and caching cohort TTL offsets...")
    build_and_save_ttl_cache()
