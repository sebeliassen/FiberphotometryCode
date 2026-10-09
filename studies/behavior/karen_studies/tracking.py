from __future__ import annotations
import pandas as pd
from pathlib import Path
import re

from syllable_generation.loaders.jonathan.utils.files import (
    parse_video_path, parse_h5_path, group_by_session_id, sort_parsed_h5_key
)
from syllable_generation.datamodel import SessionRecord
from config import JCFG

def parse_ids_from_filename(filename: str):
    parts = filename.split("_")
    for p in parts:
        if "." in p and re.search(r"\d", p):
            return [x.strip() for x in p.split(".")]
    return []

def discover_sessions_hybrid(ctx, sessions):
    """
    Project-specific non-duplicating discovery logic for karen_project:
    Groups by timestamp (ts) to avoid duplicates across different camera video files.
    """
    print("[DEBUG] [karen_project] Running custom discover_sessions_hybrid...")
    
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

    # Group by timestamp (trial level) to prevent duplicates across video files of the same session
    video_groups = {}
    for v in vids:
        video_groups.setdefault(v["ts"], []).append(v)
    print(f"[DEBUG] Found {len(vids)} videos in {len(video_groups)} trial sessions.")

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
    for ts, v_items in sorted(video_groups.items(), key=lambda kv: kv[0]):
        first_v = v_items[0]
        v_name = first_v["path"].name
        expected_ids = parse_ids_from_filename(v_name)
        
        # Determine Chambers to Generate from all video items for this timestamp
        chambers_to_make = set()
        for it in v_items:
            camset = (it["camset"] or "").lower()
            if camset in JCFG.camera_token_map:
                chambers_to_make.update(JCFG.camera_token_map[camset].values())
            elif camset == "centertopcamvid":
                chambers_to_make.update(JCFG.canon_chambers)
            else:
                if it.get("chamber") and it["chamber"] != "unknown_chamber":
                    chambers_to_make.add(it["chamber"])

        # Create Record for each chamber
        for chamber in sorted(chambers_to_make):
            # --- FILTER EMPTY ---
            if expected_ids:
                idx = JCFG.chamber_to_index.get(chamber)
                if idx is not None and idx < len(expected_ids):
                    eid = expected_ids[idx]
                    if eid.lower() == 'e':
                        print(f"[DEBUG] Skipping chamber {chamber} for trial {ts} (marked 'e' in filename)")
                        continue

            real_sid = f"{ts}_{chamber}"
            
            # Find the video that actually maps to this chamber (if any), fallback to first_v
            chamber_v = next((it for it in v_items if it.get("chamber") == chamber), first_v)
            
            meta = {
                "timestamp": ts,
                "chamber": chamber,
                "camsets_present": sorted({it["camset"] for it in v_items}),
                "cam_slots_present": sorted({it["cam_slot"] for it in v_items}),
                "arenas_present": sorted({it["arena"] for it in v_items}),
                "chamber_index": JCFG.chamber_to_index.get(chamber),
                "original_video_path": str(chamber_v["path"])
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
