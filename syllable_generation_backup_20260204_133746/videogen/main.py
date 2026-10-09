"""
syllable_generation/videogen/main.py

Entry point for the Grid Video Generator.
Produces composite videos visualizing Keypoint MoSeq syllables, Behavior Video, and Photometry Signals.

Usage:
    python syllable_generation/videogen/main.py --pkl <path> --output <dir> [options]

Key Features:
    - Multi-Panel Layouts: Duo (Side-by-Side) or Trio (Syllable | Video | Photometry)
    - Project-Aware: Automatically resolves assets (trajectory plots) based on session project.
    - Test Mode: Auto-detects interesting windows with frequent syllable transitions and available PDF plots.
    - Verification Tools: `--show-tracking` and `--flip-heading` for validating data alignment.
"""

import argparse
import pickle
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import os

# Append CWD to path to allow running as script
sys.path.append(os.getcwd())

from syllable_generation.videogen import composer
from syllable_generation.videogen import config

def load_sessions(pkl_path: str):
    print(f"Loading sessions from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        sessions = pickle.load(f)
    print(f"Loaded {len(sessions)} sessions.")
    return sessions

def find_most_active_window(session, window_size: int, pdf_dir: Optional[Path] = None) -> int:
    """
    Finds the start frame of the window with the most syllable changes,
    weighted by the existence of pre-generated trajectory PDF plots.
    """
    df = session.all_data_df
    if 'syllable' not in df.columns:
        raise ValueError("Session dataframe missing 'syllable' column.")

    # 1. Calculate Changes (transitions)
    changes = (df['syllable'] != df['syllable'].shift(1)).fillna(False).astype(int)
    
    # 2. PDF Coverage Check (Map Syllable ID -> Boolean Exists)
    pdf_exists_map = {}
    if pdf_dir and pdf_dir.exists():
        unique_sylls = df['syllable'].dropna().unique()
        for s in unique_sylls:
            s_id = int(s)
            pdf_exists_map[s_id] = (pdf_dir / f"Syllable{s_id}.pdf").exists()
    
    # 3. Calculate Score
    # Score = Transition * PDF_Availability. We want windows with many transitions to visible syllables.
    pdf_score = df['syllable'].map(lambda x: 1 if not pd.isna(x) and pdf_exists_map.get(int(x), False) else 0).fillna(0)
    valid_changes = changes * pdf_score 
    
    # 4. Rolling Sum (Find most active window)
    rolling_score = valid_changes.rolling(window=window_size).sum()
    
    # Get index of max
    max_idx = rolling_score.idxmax()
    if pd.isna(max_idx):
        return 0
        
    start_frame = int(max_idx - window_size + 1)
    return max(0, start_frame)


def main():
    parser = argparse.ArgumentParser(description="Grid Video Generator (Video + Syllables)")
    parser.add_argument("--pkl", type=str, required=True, help="Path to sessions pickle file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--session", type=int, default=0, help="Session index to process (default 0)")
    parser.add_argument("--frames", type=int, default=1000, help="Number of frames to generate")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--test-syllable-mode", action="store_true", help="Auto-detect active syllable window")
    parser.add_argument("--photometry-only", action="store_true", help="Generate only the photometry grid traces")
    parser.add_argument("--trio", action="store_true", help="Generate 3-panel layout (Syllable | Video | Photometry)")
    parser.add_argument("--show-tracking", action="store_true", help="Superimpose centroid and orientation (angle) on video")
    parser.add_argument("--flip-heading", action="store_true", help="Rotate heading by 180 degrees (pi radians) for tracking overlay")
    
    args = parser.parse_args()
    
    # Load Data
    sessions = load_sessions(args.pkl)
    if not sessions:
        print("No sessions found.")
        return

    # Apply Configuration Overrides
    if args.flip_heading:
        print("Override: Flipping tracking heading by 180 degrees.")
        config.TRACKING_HEADING_OFFSET = np.pi # 180 degrees

    try:
        session = sessions[args.session]
    except IndexError:
        print(f"Session index {args.session} out of range (0-{len(sessions)-1})")
        return

    # Resolve Video Path
    if not hasattr(session, 'behavior_video_path') or not session.behavior_video_path:
        # Heuristic: Try standard naming then check video_refs_df
        session_id = session.session_id
        candidate_name = f"Arena1234_CenterTopCamVid_{session_id}.avi"
        candidate_path = config.BEHAVIOR_VIDEO_DIR / candidate_name
        
        if not candidate_path.exists() and hasattr(session, 'video_refs_df') and session.video_refs_df is not None:
             vrefs = session.video_refs_df
             if 'path' in vrefs.columns and len(vrefs) > 0:
                 potential_path = Path(vrefs['path'].iloc[0])
                 if potential_path.exists():
                     candidate_path = potential_path

        if candidate_path.exists():
            session.behavior_video_path = candidate_path
            print(f"Resolved video path: {candidate_path}")
        else:
            print(f"Warning: Video file not found for {session_id}. Please ensure it is accessible.")
            # Continue, video grid will render black frames
    else:
        print(f"Using existing video path: {session.behavior_video_path}")

    # Logic for Test Mode
    start_frame = args.start
    num_frames = args.frames
    
    if args.test_syllable_mode:
        print(f"--- TEST SYLLABLE MODE ---")
        
        # Resolve PDF dir for prioritization
        pdf_dir = config.get_syllable_pdf_dir(getattr(session, "project", None))
        print(f"Prioritizing windows with PDFs from: {pdf_dir}")
        
        print(f"Scanning Session {args.session} for {num_frames}-frame window with most syllable transitions (weighted by PDF availability)...")
        start_frame = find_most_active_window(session, num_frames, pdf_dir=pdf_dir)
        print(f"Found active window starting at frame {start_frame}")
        
        # Print Sequence
        window = session.all_data_df.iloc[start_frame : start_frame + num_frames]
        # Iterate and print changes
        last_syll = None
        seq_str = []
        for idx, row in window.iterrows():
            current_syll = row['syllable']
            # Treat NaN
            if pd.isna(current_syll):
                val = "NaN"
            else:
                val = int(current_syll)
                
            if val != last_syll:
                seq_str.append(f"Fr{idx}:{val}")
                last_syll = val
        
        print(f"Expected Sequence: {' -> '.join(seq_str)}")
        print("--------------------------")

    end_frame = start_frame + num_frames
    
    if args.photometry_only:
        output_filename = f"Session{args.session}_F{start_frame}-{end_frame}_PhotoOnly.mp4"
        output_path = Path(args.output) / output_filename
        print(f"Generating PHOTOMETRY video for Frame {start_frame} to {end_frame}...")
        composer.generate_photometry_only_video(session, output_path, start_frame, end_frame)
    elif args.trio:
        output_filename = f"Session{args.session}_F{start_frame}-{end_frame}_Trio.mp4"
        if args.show_tracking:
            output_filename = output_filename.replace(".mp4", "_Tracking.mp4")
        output_path = Path(args.output) / output_filename
        print(f"Generating TRIO video for Frame {start_frame} to {end_frame}...")
        composer.generate_trio_video(session, output_path, start_frame, end_frame, show_tracking=args.show_tracking)
    else:
        output_filename = f"Session{args.session}_F{start_frame}-{end_frame}_Grid.mp4"
        if args.show_tracking:
            output_filename = output_filename.replace(".mp4", "_Tracking.mp4")
        output_path = Path(args.output) / output_filename
        print(f"Generating COMPOSITE video for Frame {start_frame} to {end_frame}...")
        composer.generate_session_video(session, output_path, start_frame, end_frame, show_tracking=args.show_tracking)

if __name__ == "__main__":
    main()
