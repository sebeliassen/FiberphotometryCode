"""
syllable_generation/videogen/grids/video.py

Generates the behavior video grid.
Reads frames from an OpenCV VideoCapture and formats them for the composite layout.
Support optional tracking overlays (centroid/heading) for validation.
"""

import cv2
import numpy as np
import pandas as pd
from typing import Optional
from .. import config

def get_video_grid(
    cap: cv2.VideoCapture, 
    frame_idx: int, 
    tracking_data: Optional[pd.Series] = None
) -> np.ndarray:
    """
    Retrieves the next frame from the video capture, resizes it, and optionally draws overlays.
    
    Args:
        cap: Open cv2.VideoCapture object. Caller must ensure it is positioned at the correct frame.
        frame_idx: Current frame index (used for logging or verification, not seeking).
        tracking_data: Optional row from all_data_df containing 'centroid_x', 'centroid_y', 'heading'.
    
    Returns:
        np.ndarray: Resized BGR image of size (QUADRANT_WIDTH, QUADRANT_HEIGHT).
    """
    if not cap.isOpened():
        return _make_black_frame()

    # Note: We rely on the caller (composer.py) to have set the cap position via cap.set() 
    # or sequential reads. We just read 'next'.
    
    ret, frame = cap.read()
    if not ret or frame is None:
        return _make_black_frame()

    # Draw overlays on the raw high-res frame for maximum precision
    if tracking_data is not None:
        frame = draw_tracking_overlay(frame, tracking_data)

    # Resize to quadrant size
    resized = cv2.resize(frame, (config.QUADRANT_WIDTH, config.QUADRANT_HEIGHT))
    return resized

def draw_tracking_overlay(frame: np.ndarray, row: dict, is_flipped_video: bool = False) -> np.ndarray:
    """Draws centroid and heading arrow on the frame.
    Raw coordinates are drawn directly without mathematical mirroring, since orientation mapping
    is pre-computed and inherently proven in the data model.
    """
    cx = row.get('centroid_x')
    cy = row.get('centroid_y')
    heading = row.get('heading') # assumed in radians

    if pd.isna(cx) or pd.isna(cy):
        return frame

    # Draw Centroid
    center = (int(cx), int(cy))
    # Red for original, Blue for flipped
    color = (255, 0, 0) if is_flipped_video else (0, 0, 255)
    cv2.circle(frame, center, 8, color, -1)

    # Draw Heading Arrow
    if not pd.isna(heading):
        # We add the global configured offset 
        heading += getattr(config, 'TRACKING_HEADING_OFFSET', np.pi)
        
        length = 60
        target_x = cx + length * np.cos(heading)
        target_y = cy + length * np.sin(heading) 
        
        target = (int(target_x), int(target_y))
        
        # Cyan for flipped, Green for original
        arrow_color = (255, 255, 0) if is_flipped_video else (0, 255, 0)
        cv2.arrowedLine(frame, center, target, arrow_color, 3, tipLength=0.3) 

    return frame

def _make_black_frame() -> np.ndarray:
    return np.zeros((config.QUADRANT_HEIGHT, config.QUADRANT_WIDTH, 3), dtype=np.uint8)

def resize_and_pad(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Resizes frame to fit within target dims, padding with black.
    """
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Center
    x_off = (target_w - nw) // 2
    y_off = (target_h - nh) // 2
    
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    return canvas
