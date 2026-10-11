"""
Render 9 Spliced Exemplar Videos (4 x 15s per Vibe) - Clean Timestamp Seeking
Located at: syllable_generation/analysis/language_models/pipelines/11_render_9vibe_spliced_exemplars.py

Renders 9 MP4 videos (vibe_0_spliced_60s.mp4 ... vibe_8_spliced_60s.mp4).
Each video concatenates 4 x 15-second exemplar clips sampled from 4 DIFFERENT sessions/mice.

Fixed Timestamp Conversion: Converts 80.025 Hz tracking frame indices to exact 60.0 FPS video frame offsets!
HUD Display: Strict numeric IDs only ("MACRO VIBE: VIBE X"). No descriptive text labels.
"""

import os
import pickle
import sys
import urllib.request

import cv2
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from studies.behavior.language_models.videogen.generator import get_session_video_path
from studies.behavior.language_models.videogen.overlay import draw_vibe_hud_overlay

OUT_DIR = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
VIDEO_OUT_DIR = os.path.join(OUT_DIR, "videos/vibe_spliced_60s")
NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def send_notification(msg):
    try:
        req = urllib.request.Request(NTFY_URL, data=msg.encode('utf-8'))
        with urllib.request.urlopen(req) as resp:
            pass
    except Exception:
        pass

def select_4_diverse_clips(bout_df, vibe_id):
    v_df = bout_df[bout_df["vibe_id"] == vibe_id].copy()
    v_df = v_df.dropna(subset=["cosine_sim_to_prototype"]) if "cosine_sim_to_prototype" in v_df.columns else v_df
    if "cosine_sim_to_prototype" in v_df.columns:
        v_df = v_df.sort_values("cosine_sim_to_prototype", ascending=False)

    selected = []
    seen_sessions = set()

    for _, row in v_df.iterrows():
        sid = row["session_id"]
        base_sid = sid.replace("_pre", "").replace("_post", "")
        dur = row["end_time_sec"] - row["start_time_sec"] if "end_time_sec" in row else 15.0
        if base_sid not in seen_sessions and dur >= 10.0:
            seen_sessions.add(base_sid)
            selected.append(row)
            if len(selected) >= 4:
                break

    if len(selected) < 4:
        for _, row in v_df.iterrows():
            if len(selected) >= 4:
                break
            selected.append(row)

    return pd.DataFrame(selected)

def render_spliced_video_for_vibe(vibe_id, selected_bouts_df, session_lookup):
    os.makedirs(VIDEO_OUT_DIR, exist_ok=True)
    out_mp4_path = os.path.join(VIDEO_OUT_DIR, f"vibe_{vibe_id}_spliced_60s.mp4")

    fps_out = 30
    clip_dur_sec = 15.0

    first_row = selected_bouts_df.iloc[0]
    first_sid = first_row["session_id"]
    first_sess = session_lookup.get(first_sid) or session_lookup.get(first_sid.replace("_pre", "").replace("_post", ""))
    first_vid_path = get_session_video_path(first_sess)

    cap_test = cv2.VideoCapture(first_vid_path)
    frame_width = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_test.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_mp4_path, fourcc, fps_out, (frame_width, frame_height))

    print(f"=== Rendering Vibe {vibe_id} Spliced 60s Video (4 clips x 15s) ===")

    for clip_i, (_, bout_row) in enumerate(selected_bouts_df.iterrows(), start=1):
        sid = bout_row["session_id"]
        session = session_lookup.get(sid) or session_lookup.get(sid.replace("_pre", "").replace("_post", ""))
        vid_path = get_session_video_path(session)
        df = session.all_data_df

        cap = cv2.VideoCapture(vid_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 60.0

        start_tracking_frame = int(bout_row["start_frame"])
        # Timestamp conversion: tracking frame at 80.025 Hz -> seconds -> video frame at 60.0 FPS
        start_time_sec = start_tracking_frame / 80.025
        actual_vid_frame = int(start_time_sec * video_fps)

        max_vid_frames = int(clip_dur_sec * video_fps)
        end_vid_frame = actual_vid_frame + max_vid_frames

        # Load DLC keypoints if available
        df_h5 = None
        scorer_name = None
        if hasattr(session, "tracking_refs_df") and session.tracking_refs_df is not None and not session.tracking_refs_df.empty:
            h5_path = str(session.tracking_refs_df.iloc[0]["path"])
            if os.path.exists(h5_path):
                try:
                    df_h5 = pd.read_hdf(h5_path)
                    scorer_name = df_h5.columns.get_level_values("scorer")[0]
                except Exception:
                    pass

        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_vid_frame)

        frame_stride = max(1, int(round(video_fps / fps_out)))
        curr_vid_idx = actual_vid_frame
        clip_rendered = 0

        while cap.isOpened() and curr_vid_idx < end_vid_frame:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                break

            if (curr_vid_idx - actual_vid_frame) % frame_stride == 0:
                # Map current video frame index back to 80.025 Hz tracking frame index
                t_curr_sec = curr_vid_idx / video_fps
                tracking_idx = int(t_curr_sec * 80.025)

                keypoint_coords = {}
                if df_h5 is not None and tracking_idx < len(df_h5):
                    frame_row = df_h5.iloc[tracking_idx]
                    bodyparts_list = df_h5.columns.get_level_values("bodyparts").unique()
                    for bp in bodyparts_list:
                        x = float(frame_row[(scorer_name, bp, "x")])
                        y = float(frame_row[(scorer_name, bp, "y")])
                        p = float(frame_row[(scorer_name, bp, "likelihood")]) if (scorer_name, bp, "likelihood") in frame_row else 1.0
                        keypoint_coords[bp] = (x, y, p)

                if tracking_idx < len(df):
                    row_data = df.iloc[tracking_idx]
                    syll_raw = row_data.get("syllable", -1)
                    syll_id = int(syll_raw) if pd.notna(syll_raw) else -1
                    cx = float(row_data["centroid_x"]) if ("centroid_x" in row_data and pd.notna(row_data["centroid_x"])) else None
                    cy = float(row_data["centroid_y"]) if ("centroid_y" in row_data and pd.notna(row_data["centroid_y"])) else None
                    heading = float(row_data["heading"]) if ("heading" in row_data and pd.notna(row_data["heading"])) else None
                    centroid = (cx, cy) if cx is not None and cy is not None else None
                else:
                    syll_id = -1
                    centroid = None
                    heading = None

                rendered_frame = draw_vibe_hud_overlay(
                    frame=raw_frame,
                    vibe_id=vibe_id,
                    syllable_id=syll_id,
                    vibe_names=None,
                    syllable_names=None,
                    keypoint_coords=keypoint_coords if keypoint_coords else None,
                    centroid=centroid,
                    heading=heading,
                    show_skeleton=True
                )

                h, w = rendered_frame.shape[:2]
                cv2.putText(
                    rendered_frame, f"EXEMPLAR CLIP {clip_i}/4", (w - 190, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )

                writer.write(rendered_frame)
                clip_rendered += 1

            curr_vid_idx += 1

        cap.release()
        print(f"  Clip {clip_i}/4 ({sid[:30]}) -> rendered {clip_rendered} frames")

    writer.release()
    print(f"✅ Finished Vibe {vibe_id} Spliced Video: {out_mp4_path}\n")
    return out_mp4_path

def main():
    send_notification("🎥 Starting Clean 9-Vibe Spliced Exemplar Video Rendering")

    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        sessions = pickle.load(f)

    session_lookup = {s.session_id: s for s in sessions}

    intervals_csv = os.path.join(OUT_DIR, "session_strategy_intervals.csv")
    if not os.path.exists(intervals_csv):
        print(f"Error: {intervals_csv} not found.")
        return

    bout_df = pd.read_csv(intervals_csv)

    # Load bout representativeness if available
    bout_rep_csv = os.path.join(OUT_DIR, "vibe_bout_representativeness.csv")
    if os.path.exists(bout_rep_csv):
        try:
            bout_df = pd.read_csv(bout_rep_csv)
        except Exception:
            pass

    print("=" * 70)
    print("RENDERING CLEAN 9 SPLICED EXEMPLAR VIDEOS (4 x 15s clips per Vibe)")
    print("=" * 70)

    for vibe_id in range(9):
        selected_clips = select_4_diverse_clips(bout_df, vibe_id)
        if len(selected_clips) == 0:
            print(f"Skipping Vibe {vibe_id}: No valid bouts found.")
            continue

        render_spliced_video_for_vibe(vibe_id, selected_clips, session_lookup)

    send_notification("✅ Finished rendering all 9 Clean Vibe Spliced Exemplar Videos!")
    print("=" * 70)
    print("ALL CLEAN VIBE VIDEOS RENDERED SUCCESSFULLY!")
    print(f"Videos saved to: {VIDEO_OUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
