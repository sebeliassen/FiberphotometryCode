"""
core.pair_matching
==================
Cross-camera matching, timestamp alignment, and chamber spatial mapping
for TopCam (dorsal) and BotCam (ventral) video tracking files.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .schemas import CHAMBER_MAP, PathConfig

TS_REGEX = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})")

def find_matched_pairs(
    top_h5_dir: Optional[Path] = None,
    bot_prod_dir: Optional[Path] = None
) -> List[Dict]:
    """
    Scans TopCam and BotCam directories, matching pairs by timestamp and chamber quadrant.
    Returns a list of matched session dictionaries.
    """
    paths = PathConfig()
    top_dir = Path(top_h5_dir) if top_h5_dir else paths.top_h5_dir
    bot_dir = Path(bot_prod_dir) if bot_prod_dir else paths.bot_prod_dir

    if not top_dir.exists():
        raise FileNotFoundError(f"TopCam directory does not exist: {top_dir}")
    if not bot_dir.exists():
        raise FileNotFoundError(f"BotCam directory does not exist: {bot_dir}")

    top_files = sorted(top_dir.glob("*.h5"))
    bot_dirs = sorted([d for d in bot_dir.iterdir() if d.is_dir()])

    pairs_by_sid: Dict[str, Dict] = {}
    for bdir in bot_dirs:
        h5_candidates = list(bdir.glob("*snapshot_best-180.h5")) or list(bdir.glob("*.h5"))
        if not h5_candidates:
            continue
        bot_h5 = h5_candidates[0]

        m_ts = TS_REGEX.search(bdir.name)
        if not m_ts:
            continue
        ts = m_ts.group(1)

        arena = None
        for a in ["Arena1", "Arena2", "Arena3"]:
            if a in bdir.name:
                arena = a
                break
        if not arena:
            continue
        quadrant = CHAMBER_MAP.get(arena)

        matching_top = None
        for tf in top_files:
            if ts in tf.name and quadrant in tf.name:
                matching_top = tf
                break

        if matching_top:
            sid = f"{ts}_{quadrant}"
            # Prefer canonical directory (without '_2')
            if sid not in pairs_by_sid or ("_2" not in bdir.name and "_2" in pairs_by_sid[sid]["bot_h5"].parent.name):
                pairs_by_sid[sid] = {
                    "sid": sid,
                    "ts": ts,
                    "arena": arena,
                    "quadrant": quadrant,
                    "top_h5": matching_top,
                    "bot_h5": bot_h5
                }

    return list(pairs_by_sid.values())

def circular_correlation(theta1: np.ndarray, theta2: np.ndarray) -> float:
    """
    Computes Jammalamadaka-Sarma circular-circular correlation coefficient
    between two angular arrays in radians.
    """
    valid = ~np.isnan(theta1) & ~np.isnan(theta2)
    t1 = theta1[valid]
    t2 = theta2[valid]
    if len(t1) < 100:
        return float(np.nan)

    sin1 = np.sin(t1)
    cos1 = np.cos(t1)
    mean1 = np.arctan2(np.mean(sin1), np.mean(cos1))

    sin2 = np.sin(t2)
    cos2 = np.cos(t2)
    mean2 = np.arctan2(np.mean(sin2), np.mean(cos2))

    d1 = np.sin(t1 - mean1)
    d2 = np.sin(t2 - mean2)

    num = np.sum(d1 * d2)
    den = np.sqrt(np.sum(d1**2) * np.sum(d2**2))
    return float(num / den) if den > 0 else float(np.nan)
