from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from fpcode.contracts.session_record import SessionRecord

from ..config import JCFG
from ..context import BuildContext
from ..utils.files import group_by_session_id, parse_h5_path, sort_parsed_h5_key

H5_PATTERN = re.compile(JCFG.h5_filename_regex, re.IGNORECASE)

@dataclass(frozen=True)
class ParsedH5:
    path: Path
    arena: str
    camset: str
    ts: str
    token: str
    chamber: str
    session_id: str
    cam_slot: str

def discover_sessions(ctx: BuildContext, sessions: list[SessionRecord]) -> list[SessionRecord]:
    cfg = getattr(ctx, "cfg", JCFG)
    h5_dir = ctx.h5_dir
    if not h5_dir.exists():
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")

    parsed = []
    for p in h5_dir.glob("*.h5"):  # keep glob() to preserve current behavior
        info = parse_h5_path(p, cfg)  # utils.files.parse_h5_path
        if info:
            parsed.append(info)

    groups = group_by_session_id(parsed)  # utils.files.group_by_session_id

    out: List[SessionRecord] = []
    for session_id, items in sorted(groups.items(), key=lambda kv: kv[0]):
        rows = []
        for it in sorted(items, key=sort_parsed_h5_key):  # utils.files.sort_parsed_h5_key
            rows.append({
                "path": str(it["path"]),
                "path_type": "tracking",
                "kind": "dlc_h5",
                "role": it["token"],
                "camset": it["camset"],
                "cam_slot": it["cam_slot"],
                "arena": it["arena"],
                "chamber": it["chamber"],
                "ts": it["ts"],
                "camera_key": f"{it['cam_slot']}:{it['chamber']}",
                "chamber_index": cfg.chamber_index(it["chamber"]),
            })
        tracking_refs_df = pd.DataFrame(rows)

        # safe because group keys are identical ts/chamber
        first = items[0]
        sr = SessionRecord(
            session_id=session_id,
            project=ctx.project,
            session_type="openfield",
            metadata={
                "timestamp": first["ts"],
                "chamber": first["chamber"],
                "camsets_present": sorted({it["camset"] for it in items}),
                "cam_slots_present": sorted({it["cam_slot"] for it in items}),
                "arenas_present": sorted({it["arena"] for it in items}),
                "chamber_index": cfg.chamber_to_index.get(first["chamber"]),
            },
            tracking_refs_df=tracking_refs_df,
        )
        out.append(sr)
    return out
