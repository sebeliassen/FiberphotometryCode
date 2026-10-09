"""
Vibe & Syllable Video Clip Generator
Located at syllable_generation/analysis/language_models/videogen/generator.py
"""

import os
import sys
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd

# NumPy compatibility patch for keypoint_moseq / bokeh
np.bool8 = getattr(np, 'bool8', bool)

# Path setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.language_models.videogen.overlay import draw_vibe_hud_overlay


def get_session_video_path(session) -> str:
    """
    Retrieves video file path from session object attributes.
    """
    if hasattr(session, "video_refs_df") and session.video_refs_df is not None and not session.video_refs_df.empty:
        primary_mask = session.video_refs_df["role"] == "primary"
        if primary_mask.any():
            return str(session.video_refs_df.loc[primary_mask, "path"].values[0])
        return str(session.video_refs_df.iloc[0]["path"])

    if hasattr(session, "metadata") and isinstance(session.metadata, dict):
        if "video_path" in session.metadata:
            return str(session.metadata["video_path"])

    if hasattr(session, "video_path"):
        return str(session.video_path)

    raise AttributeError(f"Could not resolve video path for session {getattr(session, 'session_id', 'unknown')}")

def find_most_active_window(df: pd.DataFrame, window_size_frames: int) -> int:
    """
    Finds the start frame of the window with maximal syllable changes and movement velocity variance.
    """
    if "syllable" not in df.columns:
        return 0

    changes = (df["syllable"] != df["syllable"].shift(1)).fillna(False).astype(float)
    vel = df["velocity_px_s"].fillna(0) if "velocity_px_s" in df.columns else pd.Series(0, index=df.index)

    score = changes * 5.0 + (vel > 10.0).astype(float) * 2.0
    rolling_score = score.rolling(window=window_size_frames).sum()

    best_end_idx = rolling_score.idxmax()
    if pd.isna(best_end_idx):
        return 0
    return max(0, int(best_end_idx - window_size_frames + 1))

def render_vibe_hud_video(
    session,
    intervals_df: pd.DataFrame,
    output_mp4_path: str,
    start_time_sec: Optional[float] = None,
    duration_sec: float = 15.0,
    fps_out: int = 30,
    show_skeleton: bool = True,
    vibe_names: Optional[Dict[int, str]] = None,
    syllable_names: Optional[Dict[int, str]] = None
) -> str:
    """
    Renders a continuous video clip with real-time Vibe & Syllable HUD overlays.
    
    Optimized OpenCV I/O: Performs a SINGLE cap.set() seek at the start,
    and reads frames sequentially to eliminate seek overhead.
    """
    os.makedirs(os.path.dirname(output_mp4_path), exist_ok=True)
    video_path = get_session_video_path(session)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at path: {video_path}")

    df = session.all_data_df
    session_id = session.session_id

    # Video capture setup
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    raw_fps = getattr(session, "primary_fps", video_fps)
    session_fps = raw_fps() if callable(raw_fps) else raw_fps
    if session_fps is None or not isinstance(session_fps, (int, float)) or session_fps <= 0:
        session_fps = 80.025

    window_size_frames = int(duration_sec * session_fps)

    # Auto-find active window if start_time_sec is None
    if start_time_sec is None:
        start_frame = find_most_active_window(df, window_size_frames)
        print(f"Auto-selected most active window start frame: {start_frame} ({start_frame/session_fps:.1f}s)")
    else:
        start_frame = int(start_time_sec * session_fps)

    # Filter intervals for this session by exact session_id or session_id + '_pre'
    exact_sid = session_id if not session_id.endswith("_pre") and not session_id.endswith("_post") else session_id
    sess_intervals = intervals_df[
        (intervals_df["session_id"] == exact_sid) |
        (intervals_df["session_id"] == f"{exact_sid}_pre") |
        (intervals_df["video_path"] == video_path)
    ]
    # Filter out _post if loading a _pre session
    if "_pre" in session_id or not session_id.endswith("_post"):
        sess_intervals = sess_intervals[~sess_intervals["session_id"].str.endswith("_post")]
    if sess_intervals.empty:
        raise ValueError(f"No strategy intervals found for session_id '{session_id}' or video_path '{video_path}'")

    # Reconstruct frame-level strategy array
    num_frames = len(df)
    frame_vibes = np.full(num_frames, -1, dtype=int)
    for _, row in sess_intervals.iterrows():
        s_frame = int(row["start_frame"])
        e_frame = int(row["end_frame"])
        v_id = int(row["strategy_id"])
        frame_vibes[s_frame:e_frame] = v_id

    end_frame = min(num_frames, start_frame + window_size_frames)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Seek ONCE at the start to prevent frame seeking overhead
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_mp4_path, fourcc, fps_out, (frame_width, frame_height))

    frame_stride = max(1, int(round(video_fps / fps_out)))

    print(f"Rendering Vibe HUD Video for Session '{session_id}'...")
    print(f"  Frames: {start_frame} -> {end_frame} ({duration_sec}s) | Stride: {frame_stride}")
    print(f"  Output MP4: {output_mp4_path}")

    curr_frame_idx = start_frame
    rendered_count = 0

    # Load DeepLabCut 27-keypoint tracking H5 if available
    df_h5 = None
    scorer_name = None
    if hasattr(session, "tracking_refs_df") and session.tracking_refs_df is not None and not session.tracking_refs_df.empty:
        h5_path = str(session.tracking_refs_df.iloc[0]["path"])
        if os.path.exists(h5_path):
            try:
                df_h5 = pd.read_hdf(h5_path)
                scorer_name = df_h5.columns.get_level_values("scorer")[0]
                print(f"Loaded DLC Keypoints from: {os.path.basename(h5_path)}")
            except Exception as e:
                print(f"Warning: Failed to load DLC H5 file {h5_path}: {e}")

    while cap.isOpened() and curr_frame_idx < end_frame:
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            break

        # Process frame if matching output fps stride
        if (curr_frame_idx - start_frame) % frame_stride == 0:
            keypoint_coords = {}
            if df_h5 is not None and curr_frame_idx < len(df_h5):
                frame_row = df_h5.iloc[curr_frame_idx]
                bodyparts_list = df_h5.columns.get_level_values("bodyparts").unique()
                for bp in bodyparts_list:
                    x = float(frame_row[(scorer_name, bp, "x")])
                    y = float(frame_row[(scorer_name, bp, "y")])
                    p = float(frame_row[(scorer_name, bp, "likelihood")]) if (scorer_name, bp, "likelihood") in frame_row else 1.0
                    keypoint_coords[bp] = (x, y, p)

            if curr_frame_idx < len(df):
                row_data = df.iloc[curr_frame_idx]
                syll_raw = row_data.get("syllable", -1)
                syll_id = int(syll_raw) if pd.notna(syll_raw) else -1
                vibe_id = int(frame_vibes[curr_frame_idx])

                cx = float(row_data["centroid_x"]) if ("centroid_x" in row_data and pd.notna(row_data["centroid_x"])) else None
                cy = float(row_data["centroid_y"]) if ("centroid_y" in row_data and pd.notna(row_data["centroid_y"])) else None
                heading = float(row_data["heading"]) if ("heading" in row_data and pd.notna(row_data["heading"])) else None
                centroid = (cx, cy) if cx is not None and cy is not None else None
            else:
                syll_id = -1
                vibe_id = -1
                centroid = None
                heading = None

            rendered_frame = draw_vibe_hud_overlay(
                frame=raw_frame,
                vibe_id=vibe_id,
                syllable_id=syll_id,
                vibe_names=vibe_names,
                syllable_names=syllable_names,
                keypoint_coords=keypoint_coords if keypoint_coords else None,
                centroid=centroid,
                heading=heading,
                show_skeleton=show_skeleton
            )

            writer.write(rendered_frame)
            rendered_count += 1

        curr_frame_idx += 1

    cap.release()
    writer.release()
    print(f"Successfully rendered {rendered_count} frames to: {output_mp4_path}")
    return output_mp4_path
