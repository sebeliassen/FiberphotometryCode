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
    print("=== Testing Vibe HUD Video Rendering ===")

    # Load session dataset pkl
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")

    print(f"Loading sessions from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        sessions = pickle.load(f)

    sample_session = sessions[0]
    print(f"Loaded sample session: {sample_session.session_id}")

    # Load 9-vibe strategy intervals CSV
    intervals_csv = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended/session_strategy_intervals.csv")
    if not os.path.exists(intervals_csv):
        raise FileNotFoundError(f"Intervals CSV not found at {intervals_csv}. Run Pipeline 1 first.")

    df_intervals = pd.read_csv(intervals_csv)

    out_video_path = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended/videos/sample_9vibe_hud.mp4")

    # Render 15-second test clip
    render_vibe_hud_video(
        session=sample_session,
        intervals_df=df_intervals,
        output_mp4_path=out_video_path,
        start_time_sec=None,  # Auto-select maximal activity window
        duration_sec=15.0,    # 15s clip
        fps_out=30,
        show_skeleton=True,
        vibe_names={
            0: "Quiescent Grooming",
            1: "Perimeter Exploration",
            2: "Center Crossing & Turning",
            3: "Fast Perimeter Run",
            5: "Repetitive Turning",
            8: "Arena Exploration"
        }
    )
    print("=== Test Video Generation Completed Successfully ===")

if __name__ == "__main__":
    main()
