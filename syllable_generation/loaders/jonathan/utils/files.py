from __future__ import annotations

import re
from pathlib import Path


def parse_h5_filename(name: str, jcfg):
    m = re.compile(jcfg.h5_filename_regex, re.IGNORECASE).match(name)
    if not m:
        return None
    arena  = m.group("arena")
    camset = m.group("camset")
    ts     = m.group("ts")
    token  = m.group("token")
    chamber   = jcfg.token_to_chamber(camset, token)
    session_id = f"{ts}_{chamber}"
    cam_slot   = jcfg.cam_slot(camset)
    return {
        "arena": arena,
        "camset": camset,
        "ts": ts,
        "token": token,
        "chamber": chamber,
        "session_id": session_id,
        "cam_slot": cam_slot,
    }

def parse_h5_path(p: Path, jcfg):
    info = parse_h5_filename(p.name, jcfg)
    if not info:
        return None
    info["path"] = p
    return info

def sort_parsed_h5_key(d):
    return (d["camset"].lower(), d["token"].lower(), str(d["path"]))

def group_by_session_id(items):
    g = {}
    for it in items:
        g.setdefault(it["session_id"], []).append(it)
    return g

def parse_video_filename(name, jcfg):
    import re
    m = re.compile(jcfg.video_filename_regex, re.I).match(name)
    if not m:
        return None
    arena  = m.group("arena")
    camset = m.group("camset")
    ts     = m.group("ts")
    token  = m.group("token")
    ext    = m.group("ext").lower()
    chamber  = jcfg.token_to_chamber(camset, token)
    cam_slot = jcfg.cam_slot(camset)
    session_id = f"{ts}_{chamber}"
    return {
        "arena": arena,
        "camset": camset,
        "ts": ts,
        "token": token,
        "ext": ext,
        "chamber": chamber,
        "cam_slot": cam_slot,
        "session_id": session_id,
    }

def parse_video_path(p, jcfg):
    info = parse_video_filename(p.name, jcfg)
    if not info:
        return None
    info["path"] = p
    return info

def gather_video_paths(video_dir):
    # keep exact behavior: flat directory, only .avi and .mp4
    return list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mp4"))

def choose_role(cam_slot):
    return "primary" if cam_slot == "center_cam" else "aux"

def sort_video_row_key(row):
    # primary first, then camset, then path (string)
    return (row["role"] != "primary", row["camset"], row["path"])
