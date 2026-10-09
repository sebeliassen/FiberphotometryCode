from __future__ import annotations

# Import LOCAL config overrides
# We assume the project folder is 'monoamine_interactions/jonathan_oft_baseline'
# and we are in 'monoamine_interactions/steps'
import sys
from pathlib import Path

import pandas as pd

from syllable_generation.datamodel import SessionRecord
from syllable_generation.loaders.jonathan.utils.files import (
    group_by_session_id,
    parse_h5_path,
    parse_video_path,
    sort_parsed_h5_key,
)

# Ensure subproject is in path to import config
subproj = Path(__file__).resolve().parents[1] / "jonathan_oft_baseline"
if str(subproj) not in sys.path:
    sys.path.insert(0, str(subproj))
from config import JCFG


def parse_ids_from_filename(filename: str):
    import re
    parts = filename.split("_")
    for p in parts:
        if "." in p and re.search(r"\d", p):
            return [x.strip() for x in p.split(".")]
    return []

def discover_sessions_hybrid(ctx, sessions):
    """
    Hybrid discovery logic:
    1. Primary discovery based on VIDEO files (ensures sessions exist even without tracking).
    2. Optional attachment of H5 tracking tracks if they exist in ctx.h5_dir.
    """
    print("[DEBUG] Running discover_sessions_hybrid...")

    video_dir = ctx.video_dir
    h5_dir = ctx.h5_dir

    # 1. Discover all Videos
    vids = []
    if video_dir and video_dir.exists():
        print(f"[DEBUG] Scanning videos in {video_dir}...")
        raw_vids = list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mp4"))
        for p in raw_vids:
            info = parse_video_path(p, JCFG)
            if info: vids.append(info)
            else: print(f"[DEBUG] Skipping unparsable video: {p.name}")

    if not vids:
        print("[DEBUG] CRITICAL: No parsable videos found!")
        return []

    video_groups = group_by_session_id(vids)
    print(f"[DEBUG] Found {len(vids)} videos in {len(video_groups)} sessions.")

    # 2. Discover all H5s (Optional)
    h5_groups = {}
    if h5_dir and h5_dir.exists():
        print(f"[DEBUG] Scanning H5s in {h5_dir}...")
        raw_h5s = list(h5_dir.glob("*.h5"))
        parsed_h5s = []
        for p in raw_h5s:
            info = parse_h5_path(p, JCFG)
            if info: parsed_h5s.append(info)
        h5_groups = group_by_session_id(parsed_h5s)
        print(f"[DEBUG] Found {len(parsed_h5s)} H5 tracking files.")

    new_sessions = []
    for sid, v_items in sorted(video_groups.items(), key=lambda kv: kv[0]):
        first_v = v_items[0]
        v_name = first_v["path"].name
        expected_ids = parse_ids_from_filename(v_name)

        # Determine Chambers to Generate
        camset = (first_v["camset"] or "").lower()
        chambers_to_make = []

        if camset in JCFG.camera_token_map:
            mapping = JCFG.camera_token_map[camset]
            chambers_to_make.extend(mapping.values())
        elif camset == "centertopcamvid":
             chambers_to_make.extend(JCFG.canon_chambers)
        else:
             # If we can't map the camera to a chamber, we used to use 'unknown_chamber'
             # But this clutters the dataset with non-animal data.
             # Only create it if there's ONLY one video and it mapped to a chamber in parse_video_path
             if first_v.get("chamber") and first_v["chamber"] != "unknown_chamber":
                 chambers_to_make.append(first_v["chamber"])
             else:
                 print(f"[DEBUG] Skipping video {v_name} - cannot map camset '{camset}' to any chamber.")
                 continue

        # Create Record for each chamber
        for chamber in chambers_to_make:
            # --- FILTER EMPTY ---
            # If we have expected IDs from the filename, check if THIS chamber is 'e'
            if expected_ids:
                idx = JCFG.chamber_to_index.get(chamber)
                if idx is not None and idx < len(expected_ids):
                    eid = expected_ids[idx]
                    if eid.lower() == 'e':
                        print(f"[DEBUG] Skipping chamber {chamber} for {sid} (marked 'e' in filename)")
                        continue

            real_sid = f"{first_v['ts']}_{chamber}"

            meta = {
                "timestamp": first_v["ts"],
                "chamber": chamber,
                "camsets_present": sorted({it["camset"] for it in v_items}),
                "cam_slots_present": sorted({it["cam_slot"] for it in v_items}),
                "arenas_present": sorted({it["arena"] for it in v_items}),
                "chamber_index": JCFG.chamber_to_index.get(chamber),
                "original_video_path": str(first_v["path"])
            }

            tracking_refs_df = pd.DataFrame()
            h_items = h5_groups.get(real_sid, [])
            if h_items:
                rows = []
                for it in sorted(h_items, key=sort_parsed_h5_key):
                    rows.append({
                        "path": str(it["path"]),
                        "path_type": "tracking",
                        "kind": "dlc_h5",
                        "role": it["token"], # upper/lower etc
                        "camset": it["camset"],
                        "cam_slot": it["cam_slot"],
                        "arena": it["arena"],
                        "chamber": it["chamber"],
                        "ts": it["ts"],
                        "camera_key": f"{it['cam_slot']}:{it['chamber']}",
                        "chamber_index": JCFG.chamber_index(it["chamber"]),
                    })
                tracking_refs_df = pd.DataFrame(rows)

            sr = SessionRecord(
                session_id=real_sid,
                project=ctx.project,
                session_type="openfield",
                metadata=meta,
                tracking_refs_df=tracking_refs_df,
            )
            new_sessions.append(sr)

    print(f"[DEBUG] Discovery complete. Created {len(new_sessions)} SessionRecord objects.")
    return new_sessions
