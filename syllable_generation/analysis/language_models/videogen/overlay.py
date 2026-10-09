"""
OpenCV Overlay Utilities for Vibe & Syllable Video Generation
Located at syllable_generation/analysis/language_models/videogen/overlay.py
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple

# Pre-defined curated vibrant color palette for 9 macro-vibes (BGR format)
VIBE_COLORS_BGR = {
    0: (235, 120, 36),   # Bright Blue
    1: (45, 180, 240),   # Warm Amber
    2: (60, 200, 80),    # Bright Green
    3: (210, 80, 180),   # Vibrant Purple
    4: (50, 215, 255),   # Bright Yellow
    5: (180, 100, 40),   # Dark Cyan
    6: (80, 90, 230),    # Bright Red
    7: (200, 160, 50),   # Steel Blue
    8: (120, 220, 140),  # Lime
}

# Pre-defined skeleton connections for mouse keypoint models
MOUSE_SKELETON_EDGES = [
    # Spine & Tail chain
    ('nose', 'head_midpoint'),
    ('head_midpoint', 'neck'),
    ('neck', 'mid_back'),
    ('mid_back', 'mouse_center'),
    ('mouse_center', 'mid_backend'),
    ('mid_backend', 'mid_backend2'),
    ('mid_backend2', 'mid_backend3'),
    ('mid_backend3', 'tail_base'),
    ('tail_base', 'tail1'),
    ('tail1', 'tail2'),
    ('tail2', 'tail3'),
    ('tail3', 'tail4'),
    ('tail4', 'tail5'),
    ('tail5', 'tail_end'),
    # Head & Ears
    ('nose', 'left_ear'),
    ('nose', 'right_ear'),
    ('left_ear', 'left_ear_tip'),
    ('right_ear', 'right_ear_tip'),
    ('head_midpoint', 'left_eye'),
    ('head_midpoint', 'right_eye'),
    # Shoulders & Hips
    ('neck', 'left_shoulder'),
    ('neck', 'right_shoulder'),
    ('left_shoulder', 'left_midside'),
    ('right_shoulder', 'right_midside'),
    ('mid_backend2', 'left_hip'),
    ('mid_backend2', 'right_hip'),
]

def draw_vibe_hud_overlay(
    frame: np.ndarray,
    vibe_id: int,
    syllable_id: int,
    vibe_names: Optional[Dict[int, str]] = None,
    syllable_names: Optional[Dict[int, str]] = None,
    keypoint_coords: Optional[Dict[str, Tuple[float, float, float]]] = None,
    centroid: Optional[Tuple[float, float]] = None,
    heading: Optional[float] = None,
    show_skeleton: bool = True
) -> np.ndarray:
    """
    Overlays real-time Vibe banner, Syllable banner, and full 27-keypoint skeleton onto video frame.
    """
    out_frame = frame.copy()
    h, w = out_frame.shape[:2]
    
    # 1. Resolve Vibe Name & Color
    vibe_name = vibe_names.get(vibe_id, f"Vibe {vibe_id}") if vibe_names else f"Vibe {vibe_id}"
    vibe_color = VIBE_COLORS_BGR.get(vibe_id % 9, (200, 200, 200))
    
    # 2. Draw Full 27-Keypoint DeepLabCut Skeleton (if keypoint_coords provided)
    if show_skeleton and keypoint_coords is not None:
        # Draw skeleton limbs
        for bp1, bp2 in MOUSE_SKELETON_EDGES:
            if bp1 in keypoint_coords and bp2 in keypoint_coords:
                x1, y1, p1 = keypoint_coords[bp1]
                x2, y2, p2 = keypoint_coords[bp2]
                if p1 >= 0.3 and p2 >= 0.3 and not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    cv2.line(out_frame, pt1, pt2, (0, 230, 100), 2, cv2.LINE_AA)
                    
        # Draw keypoint dots
        for bp, (x, y, p) in keypoint_coords.items():
            if p >= 0.3 and not (np.isnan(x) or np.isnan(y)):
                dot_color = (0, 255, 255) if 'ear' in bp or 'nose' in bp else (255, 100, 0)
                cv2.circle(out_frame, (int(x), int(y)), 3, dot_color, -1, cv2.LINE_AA)
    elif show_skeleton and centroid is not None and not np.isnan(centroid[0]) and not np.isnan(centroid[1]):
        # Fallback centroid marker
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(out_frame, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)
        if heading is not None and not np.isnan(heading):
            arrow_len = 20
            end_x = int(cx + arrow_len * np.cos(heading))
            end_y = int(cy + arrow_len * np.sin(heading))
            cv2.arrowedLine(out_frame, (cx, cy), (end_x, end_y), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
            
    # 3. Draw Compact Bottom Floating HUD Badge (keeps mouse & arena fully un-obscured)
    card_h = int(h * 0.16)
    card_top = h - card_h - 10
    card_w = int(w * 0.95)
    card_left = int((w - card_w) / 2)
    
    overlay = out_frame.copy()
    cv2.rectangle(overlay, (card_left, card_top), (card_left + card_w, card_top + card_h), (15, 15, 15), -1)
    # Color bar on the left of card
    cv2.rectangle(overlay, (card_left, card_top), (card_left + 12, card_top + card_h), vibe_color, -1)
    cv2.addWeighted(overlay, 0.75, out_frame, 0.25, 0, out_frame)
    
    vibe_text = f"MACRO VIBE: {vibe_name.upper()}"
    cv2.putText(
        out_frame, vibe_text, (card_left + 22, card_top + int(card_h * 0.45)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA
    )
    
    syll_name = syllable_names.get(syllable_id, f"Syllable #{syllable_id}") if syllable_names else f"Syllable #{syllable_id}"
    syll_text = f"MICRO SYLLABLE: {syll_name}"
    cv2.putText(
        out_frame, syll_text, (card_left + 22, card_top + int(card_h * 0.85)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA
    )
    
    return out_frame
