# video_utils.py
import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path # Added Path type hint

def resize_and_pad(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resizes a frame to fit within target dimensions while preserving aspect ratio,
    then pads with black bars to reach the exact target size.
    Returns a black frame if input is invalid or resize fails.
    """
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    if frame is None or frame.size == 0:
        return canvas

    if frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
        print(f"WARN: resize_and_pad received invalid frame shape: {frame.shape}")
        return canvas

    h, w = frame.shape[:2]

    if w <= 0 or h <= 0:
        print(f"WARN: resize_and_pad received non-positive frame dimensions: ({w}, {h})")
        return canvas

    if frame.ndim == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 1: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] != 3:
         print(f"WARN: resize_and_pad received frame with unexpected channel count: {frame.shape[2]}")
         return canvas

    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    if new_w <= 0 or new_h <= 0: return canvas

    try:
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        print(f"WARN: cv2.resize failed (shape={frame.shape}, target=({new_w},{new_h})). Error: {e}")
        return canvas

    y_off = max(0, (target_height - new_h) // 2)
    x_off = max(0, (target_width - new_w) // 2)

    h_slice = min(new_h, target_height - y_off)
    w_slice = min(new_w, target_width - x_off)

    if h_slice > 0 and w_slice > 0:
        try:
             canvas[y_off : y_off + h_slice, x_off : x_off + w_slice] = resized[0:h_slice, 0:w_slice]
        except ValueError as e:
             print(f"WARN: Error placing resized frame onto canvas. Slice mismatch? {e}")
             return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    return canvas

def open_video_capture(video_path: Path) -> Optional[cv2.VideoCapture]:
    """Safely opens a video capture object from a Path object."""
    if not video_path or not video_path.exists():
        print(f"ERROR (open_video_capture): Video path invalid or file not found: {video_path}")
        return None

    path_str = str(video_path) # Explicitly convert to string
    print(f"DEBUG (open_video_capture): Attempting cv2.VideoCapture on: '{path_str}' (Type: {type(path_str)})") # DEBUG PRINT

    cap = cv2.VideoCapture(path_str) # Use the string version

    if not cap.isOpened():
        print(f"ERROR (open_video_capture): cv2.VideoCapture FAILED to open: {path_str}")
        # Add build info here too for context if it fails
        try:
            print("    OpenCV Build Information during failure:")
            print(cv2.getBuildInformation())
        except:
            pass
        return None
    print(f"DEBUG (open_video_capture): cv2.VideoCapture SUCCEEDED for: {path_str}") # DEBUG PRINT
    return cap

def read_frame_at_index(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    """
    Reads a specific frame index from a video capture object.
    Returns None if reading fails or index is out of bounds.
    """
    if not cap or not cap.isOpened(): return None # Check if cap is valid
    # Optimization: Avoid setting frame pos if it's the next frame (doesn't always work reliably)
    # current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    # if frame_index != current_pos:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Safer to always set, though potentially slower
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def assemble_composite_frame(
    frame_tl: np.ndarray, frame_tr: np.ndarray,
    frame_bl: np.ndarray, frame_br: np.ndarray
    ) -> np.ndarray:
    """Combines four frames (assumed to be same size) into a 2x2 grid."""
    try:
        # Ensure all frames have the same height/width before stacking
        target_h = frame_tl.shape[0]
        target_w = frame_tl.shape[1]
        frames = [frame_tl, frame_tr, frame_bl, frame_br]
        # Quick check - could be more robust
        if not all(f.shape[0] == target_h and f.shape[1] == target_w for f in frames):
             print("WARN: Inconsistent frame dimensions for assembly. Returning blank.")
             return np.zeros((target_h*2, target_w*2, 3), dtype=np.uint8)

        top_row = np.hstack((frame_tl, frame_tr))
        bottom_row = np.hstack((frame_bl, frame_br))
        composite = np.vstack((top_row, bottom_row))
        return composite
    except ValueError as e:
        print(f"ERROR: Failed to stack frames. Check shapes.")
        print(f"  TL: {frame_tl.shape}, TR: {frame_tr.shape}, BL: {frame_bl.shape}, BR: {frame_br.shape}")
        h = frame_tl.shape[0] + frame_bl.shape[0] if frame_tl is not None and frame_bl is not None else 0
        w = frame_tl.shape[1] + frame_tr.shape[1] if frame_tl is not None and frame_tr is not None else 0
        return np.zeros((max(h,1), max(w,1), 3), dtype=np.uint8) # Avoid zero dim error