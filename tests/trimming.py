import numpy as np
import pandas as pd

def trim_low_likelihood(data_series, fps, threshold, trim_duration):
    """
    Given a Pandas Series of average likelihoods, mark as NaN any contiguous stretch
    that is below the threshold for at least trim_duration seconds.
    """
    below_threshold = data_series < threshold
    consecutive_frames = trim_duration * fps

    trimmed_data = data_series.copy()
    below_stretch = 0

    for i in range(len(data_series)):
        if below_threshold.iloc[i]:
            below_stretch += 1
        else:
            if below_stretch >= consecutive_frames:
                trimmed_data.iloc[i - below_stretch:i] = np.nan
            below_stretch = 0

    # Handle stretch at end
    if below_stretch >= consecutive_frames:
        trimmed_data.iloc[-below_stretch:] = np.nan

    return trimmed_data


def find_valid_segments(trimmed_series):
    """
    Given a Series with valid values and NaN for trimmed sections,
    return a list of (start, end) tuples (frame indices) for contiguous valid segments.
    """
    valid_mask = trimmed_series.notna().astype(int).values
    # Compute differences; prepend the first value so that the first frame is considered
    diff = np.diff(valid_mask, prepend=valid_mask[0])
    # 0 -> 1 transitions (diff == 1) mark the start; 1 -> 0 transitions (diff == -1) mark the end.
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0] - 1

    # Ensure the very first frame is captured if valid.
    if valid_mask[0] == 1 and (len(starts) == 0 or starts[0] != 0):
        starts = np.insert(starts, 0, 0)
    # Ensure the very last frame is captured if valid.
    if valid_mask[-1] == 1 and (len(ends) == 0 or ends[-1] != len(valid_mask) - 1):
        ends = np.append(ends, len(valid_mask) - 1)

    segments = list(zip(starts, ends))
    return segments


def choose_longest_segment(segments):
    """
    Given a list of (start, end) tuples, choose the one with the longest duration.
    If the list is empty, return None.
    """
    if segments:
        return max(segments, key=lambda s: s[1] - s[0])
    else:
        return None
    

def process_and_get_bounds_from_confidences_all(confidences, fps=60, threshold=0.8, trim_duration=15):
    """
    For each key in the confidences dictionary, compute the average likelihood per frame,
    trim stretches where the likelihood is low (for at least trim_duration seconds), and return
    all valid segments (each as a (start_frame, end_frame) tuple).
    
    Returns:
      - bounds (dict): Mapping each key to a list of (start, end) tuples.
    """
    bounds = {}
    for key, conf_array in confidences.items():
        # conf_array is assumed to have shape (n_frames, n_bodyparts)
        avg_likelihood = np.mean(conf_array, axis=1)
        avg_series = pd.Series(avg_likelihood)
        trimmed_series = trim_low_likelihood(avg_series, fps, threshold, trim_duration)
        segments = find_valid_segments(trimmed_series)
        if segments:
            bounds[key] = segments  # keep all valid segments for this key
        else:
            print(f"Warning: No valid segment found for key {key}.")
    return bounds


def create_trimmed_coords_and_confs(coordinates, confidences, errors=None):
    bounds = process_and_get_bounds_from_confidences_all(confidences, fps=60, threshold=0.8, trim_duration=15)

    trimmed_coordinates = {}
    trimmed_confidences = {}
    trimmed_errors = None if errors is None else {}
    video_frame_indexes = {}

    for key, seg_list in bounds.items():
        if len(seg_list) == 1:
            start, end = seg_list[0]
            trimmed_coordinates[key] = coordinates[key][start:end+1]
            trimmed_confidences[key] = confidences[key][start:end+1]
            if errors is not None:
                trimmed_errors[key] = errors[key][start:end+1]
            video_frame_indexes[key] = np.arange(start, end+1)
        else:
            part = 1
            for start, end in seg_list:
                seq_len = (end - start) / 60**2
                if seq_len > 5:
                    new_key = f"{key}_part{part}"
                    trimmed_coordinates[new_key] = coordinates[key][start:end+1]
                    trimmed_confidences[new_key] = confidences[key][start:end+1]
                    if errors is not None:
                        trimmed_errors[new_key] = errors[key][start:end+1]
                    video_frame_indexes[new_key] = np.arange(start, end+1)
                    part += 1

    return trimmed_coordinates, trimmed_confidences, trimmed_errors, video_frame_indexes
