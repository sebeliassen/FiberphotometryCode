import numpy as np
import pandas as pd
from typing import List

def discover_slow_syllables(sessions: List, percentile: float = 0.15) -> List[int]:
    """
    Identifies the syllable IDs with the lowest mean velocities across all sessions.
    
    Args:
        sessions: List of Session objects.
        percentile: Bottom percentile of syllables by velocity to select.
        
    Returns:
        List[int]: List of slow syllable IDs.
    """
    syllable_vels = {}
    syllable_counts = {}
    
    for s in sessions:
        df = s.all_data_df
        if "syllable" not in df.columns or "velocity_px_s" not in df.columns:
            continue
            
        # Group by syllable and calculate sum & count of velocities
        valid = df[["syllable", "velocity_px_s"]].dropna()
        valid = valid[valid["syllable"] != -1]
        
        grouped = valid.groupby("syllable")["velocity_px_s"].agg(["sum", "count"])
        for syll, row in grouped.iterrows():
            syll_id = int(syll)
            syllable_vels[syll_id] = syllable_vels.get(syll_id, 0.0) + row["sum"]
            syllable_counts[syll_id] = syllable_counts.get(syll_id, 0) + row["count"]
            
    if not syllable_vels:
        return []
        
    # Compute global mean velocity per syllable
    mean_vels = {}
    for syll_id in syllable_vels:
        if syllable_counts[syll_id] > 0:
            mean_vels[syll_id] = syllable_vels[syll_id] / syllable_counts[syll_id]
            
    # Sort syllables by mean velocity
    sorted_sylls = sorted(mean_vels.items(), key=lambda x: x[1])
    
    # Take bottom percentile of syllables
    cutoff_idx = max(1, int(len(sorted_sylls) * percentile))
    slow_syllables = [syll_id for syll_id, vel in sorted_sylls[:cutoff_idx]]
    
    return slow_syllables

def segment_by_slow_syllables(collapsed_seq: np.ndarray, slow_syll_ids: List[int]) -> List[List[int]]:
    """
    Method 1: Segments a collapsed RLE sequence into sentences using slow syllable IDs as boundaries.
    
    Args:
        collapsed_seq: Output from collapse_transitions.
        slow_syll_ids: Syllable IDs identified as slow.
        
    Returns:
        List[List[int]]: List of behavioral sentences.
    """
    sentences = []
    current_bout = []
    
    slow_set = set(slow_syll_ids)
    
    for state in collapsed_seq:
        state_val = int(state)
        if state_val in slow_set:
            if len(current_bout) > 3:
                sentences.append(current_bout)
            current_bout = []
        else:
            current_bout.append(state_val)
            
    if len(current_bout) > 3:
        sentences.append(current_bout)
        
    return sentences

def segment_by_kinematic_idleness(session, idle_vel: float = 5.0, min_duration_frames: int = 80) -> List[List[int]]:
    """
    Method 2: Slices continuous frame-by-frame velocity to locate exploration bounds,
    then extracts and collapses the corresponding RLE syllable sequences.
    
    Args:
        session: Session object with s.all_data_df.
        idle_vel: Velocity threshold below which the mouse is idle.
        min_duration_frames: Minimum contiguous frames to trigger a boundary.
        
    Returns:
        List[List[int]]: List of behavioral sentences.
    """
    df = session.all_data_df
    if "velocity_px_s" not in df.columns or "syllable" not in df.columns:
        return []
        
    vel = df["velocity_px_s"].fillna(0).to_numpy()
    syllables = df["syllable"].fillna(-1).to_numpy()
    
    is_idle = vel < idle_vel
    
    # Find start/end of contiguous idle blocks
    padded = np.concatenate(([False], is_idle, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    idle_epochs = []
    for s, e in zip(starts, ends):
        duration = e - s
        if duration >= min_duration_frames:
            idle_epochs.append((s, e))
            
    # Slicing exploration bouts (the active regions between idle epochs)
    active_bouts = []
    last_idx = 0
    
    for s, e in idle_epochs:
        if s > last_idx:
            active_bouts.append((last_idx, s))
        last_idx = e
        
    if last_idx < len(vel):
        active_bouts.append((last_idx, len(vel)))
        
    sentences = []
    for start, end in active_bouts:
        bout_sylls = syllables[start:end]
        bout_sylls = bout_sylls[bout_sylls != -1]
        
        if len(bout_sylls) == 0:
            continue
            
        # RLE collapse on the bout syllables
        mask = np.concatenate(([True], bout_sylls[1:] != bout_sylls[:-1]))
        collapsed = bout_sylls[mask].astype(int).tolist()
        
        if len(collapsed) > 3:
            sentences.append(collapsed)
            
    return sentences


def generate_sliding_windows(rich_tokens: np.ndarray, fs: float = 80.0, window_size_sec: float = 128.0, stride_sec: float = 10.0) -> List[dict]:
    """
    Slices a continuous frame-aligned array of rich tokens into overlapping sliding windows.
    
    Args:
        rich_tokens: 1D array of rich token IDs (length N).
        fs: Sampling frequency of the frames (default 80.0 Hz).
        window_size_sec: Size of each window in seconds (default 128.0s).
        stride_sec: Stride of the window in seconds (default 10.0s).
        
    Returns:
        List[dict]: List of window dictionaries with window index, bounds, and token list.
    """
    n = len(rich_tokens)
    w_size = int(window_size_sec * fs)
    stride = int(stride_sec * fs)
    
    windows = []
    if n < w_size:
        # If the session is shorter than a single window, return it as the only window
        valid_tokens = [int(x) for x in rich_tokens if x != -1]
        windows.append({
            "window_idx": 0,
            "start_frame": 0,
            "end_frame": n,
            "tokens": valid_tokens
        })
        return windows
        
    window_idx = 0
    start = 0
    while start + w_size <= n:
        end = start + w_size
        win_tokens = rich_tokens[start:end]
        valid_tokens = [int(x) for x in win_tokens if x != -1]
        
        windows.append({
            "window_idx": window_idx,
            "start_frame": start,
            "end_frame": end,
            "tokens": valid_tokens
        })
        window_idx += 1
        start += stride
        
    return windows


def assign_frame_strategies(windows_df: pd.DataFrame, num_frames: int) -> np.ndarray:
    """
    Maps window-level strategy IDs back to a continuous frame-by-frame strategy sequence.
    For any frame, assigns the strategy ID of the window whose center is closest to that frame.
    
    Args:
        windows_df: DataFrame with columns ['start_frame', 'end_frame', 'strategy_id'].
        num_frames: Total number of frames in the session.
        
    Returns:
        np.ndarray: 1D array of length num_frames containing strategy IDs.
    """
    frame_strategies = np.full(num_frames, -1, dtype=int)
    if len(windows_df) == 0:
        return frame_strategies
        
    start_frames = windows_df["start_frame"].to_numpy()
    end_frames = windows_df["end_frame"].to_numpy()
    strategy_ids = windows_df["strategy_id"].to_numpy()
    
    centers = (start_frames + end_frames) / 2.0
    frame_indices = np.arange(num_frames)
    
    # Vectorized closest center assignment
    closest_win_idx = np.clip(np.searchsorted(centers, frame_indices), 0, len(centers) - 1)
    prev_win_idx = np.clip(closest_win_idx - 1, 0, len(centers) - 1)
    
    dist_closest = np.abs(centers[closest_win_idx] - frame_indices)
    dist_prev = np.abs(centers[prev_win_idx] - frame_indices)
    
    chosen_idx = np.where(dist_closest < dist_prev, closest_win_idx, prev_win_idx)
    return strategy_ids[chosen_idx]

