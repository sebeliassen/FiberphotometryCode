import os
import pickle
import sys

import pandas as pd

# Path setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.language_models.videogen.generator import render_vibe_hud_video


def main():
    print("=== Rendering 120s Tri-Vibe Video for WT Saline Mouse ===")

    # Load session dataset pkl
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        sessions = pickle.load(f)

    # Find WT Saline session
    target_session = None
    for s in sessions:
        if "2025-04-28T14_17_17_bottom_left" in s.session_id:
            target_session = s
            break

    if target_session is None:
        raise ValueError("Could not find session 2025-04-28T14_17_17_bottom_left")

    print(f"Loaded target WT Saline session: {target_session.session_id}")

    intervals_csv = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended/session_strategy_intervals.csv")
    df_intervals = pd.read_csv(intervals_csv)

    out_video_path = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended/videos/wt_saline_trivibe_120s.mp4")

    # Render 120-second clip from 538.8s (frame 43117) to 658.8s (frame 52720)
    render_vibe_hud_video(
        session=target_session,
        intervals_df=df_intervals,
        output_mp4_path=out_video_path,
        start_time_sec=538.8,  # Start at frame 43117 (538.8s) inside Vibe 2
        duration_sec=120.0,    # 120s total duration (Vibe 2 -> Vibe 5 -> Vibe 6)
        fps_out=30,
        show_skeleton=True
    )
    print("=== Rendered 120s WT Saline Tri-Vibe Video Successfully ===")

if __name__ == "__main__":
    main()
