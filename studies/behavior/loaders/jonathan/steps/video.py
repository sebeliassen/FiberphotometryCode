# steps/video.py
from __future__ import annotations

from typing import List

import pandas as pd

from fpcode.contracts.session_record import SessionRecord

from ..config import JCFG
from ..context import BuildContext
from ..utils.files import choose_role, gather_video_paths, parse_video_path, sort_video_row_key


def attach_video_refs(ctx: BuildContext, sessions: List[SessionRecord]) -> List[SessionRecord]:
    """
    Parse video filenames, map camera tokens to canonical chamber labels,
    and attach per-session video_refs_df keyed by (timestamp × chamber).
    Also attaches fast video metadata: fps, n_frames, duration_s, meta_origin.
    """
    video_dir = ctx.video_dir or (ctx.root / "video_files")
    if not video_dir.exists():
        return sessions

    def _parse_rate(s: str):
        try:
            from fractions import Fraction
            f = Fraction(s)
            return float(f) if f.numerator and f.denominator else None
        except Exception:
            return None

    def _probe_video_fast(path):
        """
        Return (fps, n_frames, duration_s, origin) using O(1) metadata where possible.
        Order: OpenCV props -> ffprobe stream fields -> approximate.
        """
        fps = None
        n_frames = None
        duration_s = None
        origin = None

        # OpenCV first (no decoding)
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                f = cap.get(cv2.CAP_PROP_FPS)
                if f and f > 0:
                    fps = float(f)
                n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if n and n > 0:
                    n_frames = int(n)
                if fps and n_frames is not None:
                    duration_s = n_frames / fps
                origin = "opencv" if (fps or n_frames) else origin
            cap.release()
        except Exception:
            pass

        # ffprobe fallback (metadata-only)
        if fps is None or n_frames is None or duration_s is None:
            try:
                import json
                import subprocess
                cmd = [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=nb_frames,avg_frame_rate,duration",
                    "-of", "json", str(path),
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if r.returncode == 0 and r.stdout:
                    info = json.loads(r.stdout)
                    streams = info.get("streams") or []
                    if streams:
                        st = streams[0]
                        if fps is None:
                            fr = st.get("avg_frame_rate")
                            f = _parse_rate(fr) if isinstance(fr, str) else None
                            fps = f if f and f > 0 else fps
                        d = st.get("duration")
                        try:
                            d = float(d) if d is not None else None
                        except Exception:
                            d = None
                        duration_s = duration_s or d
                        nf = st.get("nb_frames")
                        try:
                            nf = int(nf) if nf is not None else None
                        except Exception:
                            nf = None
                        if n_frames is None and isinstance(nf, int) and nf > 0:
                            n_frames = nf
                        if origin is None and (fps or n_frames or duration_s):
                            origin = "ffprobe"
            except Exception:
                pass

        # last resort: approximate frame count
        if (n_frames is None) and (fps is not None) and (duration_s is not None) and duration_s > 0:
            n_frames = int(round(duration_s * fps))
            if origin is None:
                origin = "approx"

        return fps, n_frames, duration_s, origin or "unknown"

    # Parse all videos and index by session_id = f"{ts}_{chamber}"
    index: dict[str, list[dict]] = {}
    for p in gather_video_paths(video_dir):
        info = parse_video_path(p, JCFG)
        if not info:
            continue

        fps, n_frames, duration_s, origin = _probe_video_fast(p)

        row = {
            "path": str(p),
            "path_type": "video",
            "kind": info["ext"],
            "role": choose_role(info["cam_slot"]),
            "ts": info["ts"],
            "chamber": info["chamber"],
            "camset": info["camset"],
            "cam_slot": info["cam_slot"],
            "arena": info["arena"],
            "camera_key": f"{info['cam_slot']}:{info['chamber']}",
            "fps": fps,
            "n_frames": n_frames,
            "duration_s": duration_s,
            "meta_origin": origin,  # "opencv" | "ffprobe" | "approx" | "unknown"
        }
        index.setdefault(info["session_id"], []).append(row)

    # Attach to sessions
    for s in sessions:
        rows = index.get(s.session_id, [])
        if rows:
            s.video_refs_df = pd.DataFrame(sorted(rows, key=sort_video_row_key))
    return sessions
