"""Visualization legacy io; historical V2 behavior preserved."""

from __future__ import annotations

from fpcode.viz.legacy_defaults import BODYPARTS_21KP
from fpcode.viz.legacy_defaults import BOT_VID_DIR
from fpcode.viz.legacy_defaults import CHAMBER_TO_ARENA
from fpcode.viz.legacy_defaults import FUSED_DIR
from fpcode.viz.legacy_defaults import TOP_VID_DIR
from fpcode.viz.media import load_fused_tracks as read_fused_tracks
from fpcode.viz.media import media_file_identity as identify_media
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
import numpy as np
import re


def get_session_files(sid: str, pairs_by_sid: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Resolve Top/Bot videos and DLC h5 files for a (possibly segmented) session id."""
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})_([a-z]+_[a-z]+)", sid)
    if not m:
        return None
    ts, chamber = m.group(1), m.group(2)
    arena = CHAMBER_TO_ARENA.get(chamber)
    if not arena:
        return None
    pair = pairs_by_sid.get(f"{ts}_{chamber}")
    if not pair:
        return None

    top_vids = list(TOP_VID_DIR.glob(f"*{ts}*{chamber}*.avi"))
    if len(top_vids) > 1 and "top_h5" in pair:
        top_stem = re.sub(r"DLC_.*\.h5$", "", Path(pair["top_h5"]).name)
        matched = [v for v in top_vids if v.stem == top_stem or top_stem in v.name]
        if matched:
            top_vids = matched[:1]

    bot_vids = list(BOT_VID_DIR.glob(f"*{arena}*BotCamVid*{ts}*.avi"))
    if len(bot_vids) > 1 and "top_h5" in pair:
        has_two = " (2)" in Path(pair["top_h5"]).name
        matched = [v for v in bot_vids if (" (2)" in v.name) == has_two]
        if matched:
            bot_vids = matched[:1]
        else:
            bot_vids = bot_vids[:1]

    if len(top_vids) > 1 or len(bot_vids) > 1:
        raise ValueError(f"Ambiguous source videos for {sid}: top={top_vids}, bot={bot_vids}")
    if top_vids and bot_vids and Path(pair["top_h5"]).exists() and Path(pair["bot_h5"]).exists():
        return {
            "top_vid": str(top_vids[0]),
            "bot_vid": str(bot_vids[0]),
            "top_h5": str(pair["top_h5"]),
            "bot_h5": str(pair["bot_h5"]),
        }
    return None


def media_file_identity(manifest: Dict[str, Any], syllable: int) -> Dict[str, Any]:
    """Cache invalidation for selected source media/tracks (path/size/mtime)."""
    from fpcode.viz.selection.bouts import base_session_id

    artifacts = {}
    for b in manifest["syllables"][str(syllable)]["bouts"]:
        if b.get("is_blank"):
            continue
        base = base_session_id(b["segment_name"])
        artifacts[base] = FUSED_DIR / f"{base}_tandem_21kp.h5"
    return identify_media(manifest, syllable, fused_artifacts=artifacts)


def load_fused_tracks(base_sid: str, fused_dir: Path = FUSED_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the ORIGINAL fused observations for a session: ``xy (T, 21, 2)`` and
    raw likelihoods ``conf (T, 21)``, in ``BODYPARTS_21KP`` order.  Missing
    BotCam-mapped frames are NaN / likelihood 0 (see static_fusion_engine).
    """
    path = Path(fused_dir) / f"{base_sid}_tandem_21kp.h5"
    return read_fused_tracks(path, BODYPARTS_21KP)
