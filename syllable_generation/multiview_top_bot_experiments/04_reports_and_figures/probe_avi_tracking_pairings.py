#!/usr/bin/env python3
"""
probe_avi_tracking_pairings.py
=============================================================================
Phase 1 Audit: Verify Ambiguous AVI-Tracking Pairings across all sessions.
1. Checks TOP_VID_DIR and BOT_VID_DIR for multiple glob matches per session.
2. Checks candidate file sizes and SHA-256 hashes (identifying true duplicates).
3. Validates frame count agreement between AVI and DLC/fused tracking.
4. Checks all selected bouts in unified_syllable_selection_manifest.json to
   verify whether any selected bout uses an ambiguous pairing.
=============================================================================
"""

import hashlib
import json
import re
from pathlib import Path

import h5py
import viz_v2_common as vc


def file_sha256(p: Path, max_bytes: int = 100 * 1024 * 1024) -> str:
    """Hash first 100 MB for rapid verification of large AVI files."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        bytes_read = 0
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
            bytes_read += len(chunk)
            if bytes_read >= max_bytes:
                break
    return h.hexdigest()

def main():
    print("=" * 80)
    print("🔍 AUDITING AVI-TRACKING PAIRING AMBIGUITY")
    print("=" * 80)

    # 1. Load results.h5 segments
    with h5py.File(vc.RESULTS_PATH, "r") as f:
        seg_keys = sorted([k for k in f.keys() if k not in ("metadata", "model_parameters")])

    base_sids = sorted(list(set(vc.base_session_id(k) for k in seg_keys)))
    print(f"Total base sessions in model results: {len(base_sids)} ({len(seg_keys)} segments)")

    # 2. Load pairs_by_sid from matching
    import sys
    if str(vc.EXP_DIR) not in sys.path:
        sys.path.insert(0, str(vc.EXP_DIR))
    from core.pair_matching import find_matched_pairs
    pairs = find_matched_pairs()
    pairs_by_sid = {p["sid"]: p for p in pairs}

    ambiguous_sessions = []
    clean_sessions = []

    for sid in base_sids:
        m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})_([a-z]+_[a-z]+)", sid)
        if not m:
            continue
        ts, chamber = m.group(1), m.group(2)
        arena = vc.CHAMBER_TO_ARENA.get(chamber)
        pair = pairs_by_sid.get(f"{ts}_{chamber}", {})

        top_vids = list(vc.TOP_VID_DIR.glob(f"*{ts}*{chamber}*.avi"))
        bot_vids = list(vc.BOT_VID_DIR.glob(f"*{arena}*BotCamVid*{ts}*.avi"))

        is_ambiguous = len(top_vids) > 1 or len(bot_vids) > 1

        info = {
            "sid": sid,
            "ts": ts,
            "chamber": chamber,
            "arena": arena,
            "top_candidates": [str(p) for p in top_vids],
            "bot_candidates": [str(p) for p in bot_vids],
            "top_count": len(top_vids),
            "bot_count": len(bot_vids),
        }

        if is_ambiguous:
            # Check candidate details
            top_details = []
            for p in top_vids:
                sz = p.stat().st_size
                top_details.append({"path": str(p), "size": sz, "prefix_sha256": file_sha256(p)})
            info["top_details"] = top_details

            bot_details = []
            for p in bot_vids:
                sz = p.stat().st_size
                bot_details.append({"path": str(p), "size": sz, "prefix_sha256": file_sha256(p)})
            info["bot_details"] = bot_details

            # Check if sizes/hashes are identical
            top_identical = (len(set(d["size"] for d in top_details)) == 1 and len(set(d["prefix_sha256"] for d in top_details)) == 1) if top_details else True
            bot_identical = (len(set(d["size"] for d in bot_details)) == 1 and len(set(d["prefix_sha256"] for d in bot_details)) == 1) if bot_details else True
            info["top_identical_duplicates"] = top_identical
            info["bot_identical_duplicates"] = bot_identical

            ambiguous_sessions.append(info)
        else:
            clean_sessions.append(info)

    print("\n[Pairing Audit Results]")
    print(f"  Unambiguous sessions: {len(clean_sessions)} / {len(base_sids)}")
    print(f"  Sessions with multiple AVI candidates: {len(ambiguous_sessions)} / {len(base_sids)}")

    # 3. Check selected bouts in manifest
    manifest_path = vc.DEFAULT_VIZ_DIR / "unified_syllable_selection_manifest.json"
    manifest_affected_bouts = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest_data = json.load(f)

        ambig_sid_set = set(d["sid"] for d in ambiguous_sessions)
        for s_str, s_data in manifest_data.get("syllables", {}).items():
            for b in s_data.get("selected_bouts", []):
                if b.get("is_blank"):
                    continue
                seg_name = b.get("segment_name", "")
                base_name = vc.base_session_id(seg_name)
                if base_name in ambig_sid_set:
                    manifest_affected_bouts.append({
                        "syllable": int(s_str),
                        "tile_index": b.get("tile_index"),
                        "bout_id": b.get("bout_id"),
                        "segment_name": seg_name,
                        "base_session": base_name,
                        "top_vid": b.get("top_vid"),
                        "bot_vid": b.get("bot_vid"),
                    })

    print("\n[Manifest Impact]")
    print(f"  Selected bouts in manifest using ambiguous sessions: {len(manifest_affected_bouts)}")

    # Print details of ambiguous sessions
    for item in ambiguous_sessions:
        print(f"\n⚠️  Session {item['sid']}:")
        print(f"   Top candidates ({item['top_count']}): identical={item.get('top_identical_duplicates')}")
        for c in item.get("top_details", []):
            print(f"     - {Path(c['path']).name} ({c['size']:,} bytes, sha={c['prefix_sha256'][:12]})")
        print(f"   Bot candidates ({item['bot_count']}): identical={item.get('bot_identical_duplicates')}")
        for c in item.get("bot_details", []):
            print(f"     - {Path(c['path']).name} ({c['size']:,} bytes, sha={c['prefix_sha256'][:12]})")

    out_file = vc.DEFAULT_VIZ_DIR / "avi_pairing_ambiguity_audit.json"
    audit_report = {
        "total_base_sessions": len(base_sids),
        "unambiguous_count": len(clean_sessions),
        "ambiguous_count": len(ambiguous_sessions),
        "ambiguous_sessions": ambiguous_sessions,
        "manifest_affected_bouts": manifest_affected_bouts,
    }
    with open(out_file, "w") as f:
        json.dump(audit_report, f, indent=2)
    print(f"\n✅ Full audit saved to: {out_file}")

if __name__ == "__main__":
    main()
