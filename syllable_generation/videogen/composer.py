"""
syllable_generation/videogen/composer.py

Orchestrates the generation of composite videos.
Combines output from `SyllableGridGenerator`, `PhotometryGridGenerator`, and generic video frames.
Handles frame synchronization, real-speed sampling (stride calculation), and layout assembly.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..datamodel import SessionRecord
from . import config
from .grids.photometry import PhotometryGridGenerator
from .grids.syllable import SyllableGridGenerator
from .grids.video import draw_tracking_overlay, get_video_grid


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
    show_tracking: bool = False,
    pdf_dir: Optional[Path] = None
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
    syllable_gen = SyllableGridGenerator(project_name=getattr(session, "project", None), pdf_dir_override=pdf_dir)

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
    show_tracking: bool = False,
    pdf_dir: Optional[Path] = None
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
    syllable_gen = SyllableGridGenerator(project_name=session.project, pdf_dir_override=pdf_dir)
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


def generate_kinematic_verification_video(
    session: SessionRecord,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    verbose: bool = True,
    pdf_dir: Optional[Path] = None,
    vfi: Optional[np.ndarray] = None,
    snapshots_only: bool = False
):
    """
    Generates a 2x2 comparison video for Kinematic Verification.
    Layout (1600x1600): 
    [ Plot Orig (Syll X) | Plot Flip (Syll Y) ]
    [ Video Orig (Track) | Video Flip (Track-F) ]
    """
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    video_path = getattr(session, 'behavior_video_path', '')
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open behavior video for {session.session_id}")
        return

    # --- Setup Video Output ---
    if not snapshots_only:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            40, # Encode at exactly 40 FPS to match the behavioral camera and play in real-time
            (config.TRIO_WIDTH, config.TRIO_HEIGHT) # 1600x1600
        )

    # --- Setup Generators ---
    syllable_gen = SyllableGridGenerator(project_name=session.project, pdf_dir_override=pdf_dir)

    df = session.all_data_df
    total_frames_data = len(df)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, start_frame)
    end_frame = min(end_frame, min(total_frames_data, total_frames_video))

    # --- Setup Snapshots ---
    # Filter out empty subset padding or ghost EOF frames
    if 'frame_index' not in df.columns:
        print("Error: all_data_df must have 'frame_index' for direct mapping!")
        return

    valid_mask = df['centroid_x'].notna() & (df['frame_index'] < total_frames_video) & (df.index >= start_frame) & (df.index < end_frame)
    valid_rows = df[valid_mask]

    if len(valid_rows) == 0:
        print(f"Error: No valid tracking frames found in video bounds for {session.session_id}")
        return

    if snapshots_only:
        # 5 evenly spaced indices from the valid portion of the DataFrame
        rel_indices = np.linspace(0, len(valid_rows) - 1, 5, dtype=int)
        process_indices = valid_rows.index[rel_indices].tolist()
        # Map relative array position 0-4
        snap_labels = {i: f"snap_{i}" for i in range(len(process_indices))}
    else:
        # Target real-time playback by downsampling 80Hz data to 40Hz (matching camera)
        valid_rows = valid_rows.iloc[::2]

        process_indices = valid_rows.index.tolist()
        v_len = len(process_indices)
        snap_labels = {}
        if v_len > 5:
            snap_labels = {
                5: "start",
                v_len // 2: "middle",
                max(0, v_len - 5): "end"
            }

    frame_dir = output_path.parent / (output_path.stem + "_frames")
    frame_dir.mkdir(exist_ok=True)

    # Frame Loop
    prog_bar = tqdm(enumerate(process_indices), total=len(process_indices), disable=not verbose, unit="frames")

    for relative_i, df_idx in prog_bar:
        row = df.loc[df_idx]

        # The definitive truth of which video frame corresponds to this dataframe entry
        # is stored in the npz array (VFI) we loaded earlier.
        # When snapshots_only is set or when we process the slice, we need `v_idx`
        #
        # If the user requested VFI direct, we should map back via `df.index`
        # But wait - we don't have vfi_array available directly inside composer's frame loop unless we pass it!
        # It's passed via `vfi=vfi_array`. We can just use `vfi[relative_i]` ? No, relative_i is the slice index.
        # df.index is the physical index in `all_data_df` which might be densified.
        # But the problem is 1) Session 0 works with VFI direct 2) Session 5 works with 1:1.
        # For now, let's revert to the purely neutral form and let the user decide the permanent fix.
        v_idx = int(row['frame_index'])

        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_pos == v_idx:
            # We are already at the right frame, just read
            ret, frame = cap.read()
        elif current_pos < v_idx and (v_idx - current_pos) < 30:
            # If the target is just a few frames ahead, cap.read() is typically MUCH faster than cap.set()
            for _ in range(v_idx - current_pos):
                ret, frame = cap.read()
        else:
            # We have to jump or rewind
            cap.set(cv2.CAP_PROP_POS_FRAMES, v_idx)
            ret, frame = cap.read()

        if not ret or frame is None:
            if verbose: print(f"Skipping V-Frame {v_idx} because cap.read() failed")
            frame = np.zeros((config.QUADRANT_HEIGHT, config.QUADRANT_WIDTH, 3), dtype=np.uint8)

        # 1. Top Row: Trajectory Plots
        syll_orig = row.get('syllable', -1)
        syll_flipped = row.get('syllable_flipped', -1)

        frame_plot_orig = syllable_gen.get_syllable_grid(int(syll_orig) if not pd.isna(syll_orig) else -1)
        frame_plot_flipped = syllable_gen.get_syllable_grid(int(syll_flipped) if not pd.isna(syll_flipped) else -1)

        # 2. Bottom Row: Videos
        # The video is always shown in its raw form for the "Original" panel
        # and horizontally flipped for the "Flipped" panel.
        frame_actual_orig = frame.copy()
        frame_actual_flipped = cv2.flip(frame, 1)

        # Bottom Left: Original Tracking (Relative to original space)
        # We manually build a clean dictionary to pass to draw_tracking_overlay
        orig_row = {
            'centroid_x': row['centroid_x'],
            'centroid_y': row['centroid_y'],
            'heading': row['heading']
        }

        # Bottom Right: Flipped Tracking (Model's actual unmirrored prediction natively in flipped space)
        # Again, raw extraction, no mirroring needed for draw_tracking_overlay anymore
        cxf_raw = row.get('centroid_x_flipped')
        if pd.isna(cxf_raw):
            cxf_raw = row['centroid_x']
            cyf_raw = row['centroid_y']
            hf_raw = row['heading']
        else:
            cyf_raw = row['centroid_y_flipped']
            hf_raw = row['heading_flipped']

        flip_row = {
            'centroid_x': cxf_raw,
            'centroid_y': cyf_raw,
            'heading': hf_raw
        }

        # Override module config so draw_tracking correctly skips mirroring inside its own function too,
        # or we just rely on the fixed draw_tracking_overlay which now expects raw coords
        frame_orig_tracked = draw_tracking_overlay(frame_actual_orig, orig_row, is_flipped_video=False)
        frame_orig_final = cv2.resize(frame_orig_tracked, (config.QUADRANT_WIDTH, config.QUADRANT_HEIGHT))

        frame_flipped_tracked = draw_tracking_overlay(frame_actual_flipped, flip_row, is_flipped_video=True)
        frame_flipped_final = cv2.resize(frame_flipped_tracked, (config.QUADRANT_WIDTH, config.QUADRANT_HEIGHT))

        # Assemble 2x2
        top_half = np.hstack((frame_plot_orig, frame_plot_flipped))
        bottom_half = np.hstack((frame_orig_final, frame_flipped_final))
        composite = np.vstack((top_half, bottom_half))

        # Add labels
        cv2.putText(composite, "ORIGINAL", (20, config.QUADRANT_HEIGHT + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(composite, "FLIPPED", (config.QUADRANT_WIDTH + 20, config.QUADRANT_HEIGHT + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        is_snapshot = relative_i in snap_labels

        if not snapshots_only:
            writer.write(composite)

        # Save snapshots
        if is_snapshot:
            label = snap_labels[relative_i]
            fname = f"{output_path.stem}_{label}.png"
            cv2.imwrite(str(frame_dir / fname), composite)
            if verbose:
                print(f"Saved snapshot: {fname}")

    # Cleanup
    cap.release()
    if not snapshots_only:
        writer.release()
    print(f"Saved Kinematic Verification results to {output_path.parent}")
