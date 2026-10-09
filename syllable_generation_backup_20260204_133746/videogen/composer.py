"""
syllable_generation/videogen/composer.py

Orchestrates the generation of composite videos.
Combines output from `SyllableGridGenerator`, `PhotometryGridGenerator`, and generic video frames.
Handles frame synchronization, real-speed sampling (stride calculation), and layout assembly.
"""

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple

from ..datamodel import SessionRecord
from . import config
from .grids.video import get_video_grid
from .grids.syllable import SyllableGridGenerator
from .grids.photometry import PhotometryGridGenerator

def generate_photometry_only_video(
    session: SessionRecord,
    output_path: Path,
    start_frame: int,
    end_frame: int, 
    verbose: bool = True
):
    """
    Generates video containing ONLY the photometry grid.
    """
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup Output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Using COMPOSITE dimensions for consistency? 
    # Or just default 1600x800 as defined in PhotometryGridGenerator base logic (which uses config.COMPOSITE_WIDTH).
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        config.TARGET_FPS,
        (config.COMPOSITE_WIDTH, config.COMPOSITE_HEIGHT)
    )

    photo_gen = PhotometryGridGenerator()
    
    # Pre-parse signal meta (just once)
    signal_meta = getattr(session.metadata, 'get', lambda k: {})('signal_meta')
    if not signal_meta:
        # Fallback: maybe it's directly in metadata?
        signal_meta = session.metadata.get('signal_meta', {})

    # Data Window Logic
    # We need a rolling window. 
    # Let's slice the dataframe dynamically.
    df = session.all_data_df
    total_frames = len(df)
    
    start_frame = max(0, start_frame)
    end_frame = min(end_frame, total_frames)

    prog_bar = tqdm(range(start_frame, end_frame), disable=not verbose, unit="frames")
    
    half_window = config.PLOT_WINDOW_SIZE // 2

    for i in prog_bar:
        # Determine Window
        w_start = max(0, i - half_window)
        w_end = min(total_frames, i + half_window)
        
        df_window = df.iloc[w_start:w_end]
        
        # Generator expects window + current frame index (absolute or relative?)
        # My implementation assumed absolute 'current_frame_idx' to draw the vertical line.
        # But if we pass the window slice, plotting X vs Y using window index...
        # If I pass 'i' as current_frame_idx, and plot using df.index (which is absolute), 
        # then ax.axvline(x=i) works perfectly.
        
        frame = photo_gen.generate_frame(df_window, i, signal_meta)
        
        writer.write(frame)

    photo_gen.close()
    writer.release()
    print(f"Saved Photometry-Only video to {output_path}")

def generate_session_video(
    session: SessionRecord,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    verbose: bool = True,
    show_tracking: bool = False
):
    """
    Generates a composite video for the given session and frame range.
    Layout: Side-by-Side (Video Left, Syllable Right).
    """
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    
    # Resolve Behavior Video Path
    # Uses session.behavior_video_path (set by main.py) or falls back to attribute check.
    cap = cv2.VideoCapture(str(getattr(session, 'behavior_video_path', '')))
    if not cap.isOpened():
        print(f"Error: Could not open behavior video for {session.session_id}")
        return

    # --- Setup Video Output ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        config.TARGET_FPS,
        (config.COMPOSITE_WIDTH, config.COMPOSITE_HEIGHT)
    )

    # --- Setup Grid Generators ---
    syllable_gen = SyllableGridGenerator()

    # --- Frame Loop ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, start_frame)
    if end_frame < 0 or end_frame > total_frames:
        end_frame = total_frames
    
    # Clamp to Dataframe Length
    df_len = len(session.all_data_df)
    if end_frame > df_len:
        print(f"Warning: Frame range end {end_frame} exceeds dataframe length {df_len}. Clamping.")
        end_frame = df_len

    # Data Sampling (Real Speed Logic)
    # Calculates stride to map source FPS (e.g. 80Hz) to Target FPS (30Hz).
    source_fps = config.PROJECT_SAMPLING_RATES.get(session.project, config.DEFAULT_SAMPLING_RATE)
    stride = source_fps / config.TARGET_FPS
    
    num_source_frames = end_frame - start_frame
    num_output_frames = int(num_source_frames / stride)
    
    if verbose:
        print(f"Sampling real-speed video: {source_fps}Hz -> {config.TARGET_FPS}Hz (Stride: {stride:.2f})")

    # Frame Loop
    prog_bar = tqdm(range(num_output_frames), disable=not verbose, unit="frames")
    
    for i in prog_bar:
        # Calculate current source index based on stride
        curr_source_idx = start_frame + int(i * stride)
        
        # Seek video to match data index
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_source_idx)
        
        # 1. Video Grid
        tracking_data = session.all_data_df.iloc[curr_source_idx] if show_tracking else None
        frame_video = get_video_grid(cap, curr_source_idx, tracking_data=tracking_data)
        
        # 2. Syllable Grid
        try:
            syll_val = session.all_data_df['syllable'].iloc[curr_source_idx]
            # Handle Missing/NaN syllables
            if pd.isna(syll_val):
                frame_syll = syllable_gen.get_syllable_grid(-1) 
            else:
                frame_syll = syllable_gen.get_syllable_grid(int(syll_val))
        except IndexError:
             frame_syll = syllable_gen.get_syllable_grid( -999 ) # Out of bounds
        
        # 3. Composite (Side-by-Side)
        composite = np.hstack((frame_video, frame_syll))
        
        writer.write(composite)

    # Cleanup
    cap.release()
    writer.release()
    print(f"Saved video to {output_path}")


def generate_trio_video(
    session: SessionRecord,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    verbose: bool = True,
    show_tracking: bool = False
):
    """
    Generates a 3-panel composite video.
    Layout: 
    [ Syllable (L) | Video (R) ]
    [      Photometry (Full)   ]
    Total: 1600x1600
    """
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Video Path Resolution ---
    video_path = getattr(session, 'behavior_video_path', '')
    if not video_path:
        # Try to find it if we're in the right env
        pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open behavior video for {session.session_id}")
        return

    # --- Setup Video Output ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        config.TARGET_FPS,
        (config.TRIO_WIDTH, config.TRIO_HEIGHT)
    )

    # --- Setup Generators ---
    syllable_gen = SyllableGridGenerator(project_name=session.project)
    photo_gen = PhotometryGridGenerator()
    
    # Pre-parse meta
    signal_meta = getattr(session.metadata, 'get', lambda k: {})('signal_meta')
    if not signal_meta:
        signal_meta = session.metadata.get('signal_meta', {})

    # Data Window Logic
    df = session.all_data_df
    total_frames_data = len(df)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = max(0, start_frame)
    end_frame = min(end_frame, min(total_frames_data, total_frames_video))

    # --- Sampling Logic for Real Speed ---
    source_fps = config.PROJECT_SAMPLING_RATES.get(session.project, config.DEFAULT_SAMPLING_RATE)
    stride = source_fps / config.TARGET_FPS
    
    num_source_frames = end_frame - start_frame
    num_output_frames = int(num_source_frames / stride)

    if verbose:
        print(f"Sampling real-speed trio video: {source_fps}Hz -> {config.TARGET_FPS}Hz (Stride: {stride:.2f})")
        print(f"Output will be {num_output_frames} frames.")

    prog_bar = tqdm(range(num_output_frames), disable=not verbose, unit="frames")
    
    half_window = config.PLOT_WINDOW_SIZE // 2

    for i in prog_bar:
        curr_source_idx = start_frame + int(i * stride)
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_source_idx)

        # 1. Video Grid (Top Right)
        tracking_data = df.iloc[curr_source_idx] if show_tracking else None
        frame_video = get_video_grid(cap, curr_source_idx, tracking_data=tracking_data)
        
        # 2. Syllable Grid (Top Left)
        try:
            syll_val = df['syllable'].iloc[curr_source_idx]
            if pd.isna(syll_val):
                frame_syll = syllable_gen.get_syllable_grid(-1)
            else:
                frame_syll = syllable_gen.get_syllable_grid(int(syll_val))
        except:
            frame_syll = syllable_gen.get_syllable_grid(-999)

        # 3. Photometry Grid (Bottom)
        w_start = max(0, curr_source_idx - half_window)
        w_end = min(total_frames_data, curr_source_idx + half_window)
        df_window = df.iloc[w_start:w_end]
        frame_photo = photo_gen.generate_frame(df_window, curr_source_idx, signal_meta, is_trio=True)

        # --- Assemble ---
        # Top half: [Syllable | Video]
        top_half = np.hstack((frame_syll, frame_video))
        # Bottom half: Photometry (already 1600 wide)
        composite = np.vstack((top_half, frame_photo))
        
        writer.write(composite)

    # Cleanup
    cap.release()
    writer.release()
    photo_gen.close()
    print(f"Saved Trio video to {output_path}")
