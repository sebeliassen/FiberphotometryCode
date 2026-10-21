"""Pose coordinates and historical keypoint layouts shared across domains."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .timelines import TimeAxis, frozen_array

# TopCam (Dorsal) 17 keypoints
TOP_USE_BODYPARTS: List[str] = [
    "nose",
    "left_ear_tip",
    "right_ear_tip",
    "neck",
    "mid_back",
    "mouse_center",
    "mid_backend",
    "mid_backend2",
    "mid_backend3",
    "tail_base",
    "left_shoulder",
    "left_midside",
    "left_hip",
    "right_shoulder",
    "right_midside",
    "right_hip",
    "head_midpoint",
]

# BotCam (Ventral) canonical tracking names
BOT_RAW_BODYPARTS: List[str] = [
    "left_front_paw",
    "right_front_paw",
    "left_hind_paw",
    "right_hind_paw",
    "nose_tip",
    "tail_base",
]

# Mapping to non-conflicting names in the fused tandem space
BOT_RENAME_MAP: Dict[str, str] = {
    "left_front_paw": "left_front_paw",
    "right_front_paw": "right_front_paw",
    "left_hind_paw": "left_hind_paw",
    "right_hind_paw": "right_hind_paw",
    "nose_tip": "bot_nose_tip",
    "tail_base": "bot_tail_base",
}

BOT_USE_BODYPARTS: List[str] = list(BOT_RENAME_MAP.values())

# Backward-compatible and clean schema aliases
DORSAL_17KP_NAMES: List[str] = TOP_USE_BODYPARTS
VENTRAL_4KP_NAMES: List[str] = [
    "left_front_paw",
    "right_front_paw",
    "left_hind_paw",
    "right_hind_paw",
]
FUSED_21KP_NAMES: List[str] = TOP_USE_BODYPARTS + VENTRAL_4KP_NAMES

# Combined Tandem 23 Keypoints (17 TopCam + 6 BotCam)
TANDEM_BODYPARTS: List[str] = TOP_USE_BODYPARTS + BOT_USE_BODYPARTS
TANDEM_23KP: List[str] = TANDEM_BODYPARTS
FUSED_23KP_NAMES: List[str] = TANDEM_23KP

BODYPART_REMAP: Dict[str, str] = {
    "snout": "nose_ventral",
    "left_ear": "left_ear_ventral",
    "right_ear": "right_ear_ventral",
    "tailbase": "tail_base_ventral",
}

# -----------------------------------------------------------------------------
# 2. Chamber & Spatial Alignment Schemas
# -----------------------------------------------------------------------------

CHAMBER_MAP: Dict[str, str] = {
    "Arena1": "top_left",
    "Arena2": "bottom_left",
    "Arena3": "bottom_right",
    "Arena4": "top_right",
}

DEFAULT_ARENA_WIDTH: float = 972.0
DEFAULT_TARGET_FPS: int = 60


@dataclass(frozen=True)
class PoseSeries:
    time: TimeAxis
    coordinates: np.ndarray
    confidence: np.ndarray
    bodyparts: Tuple[str, ...]
    unit: str
    coordinate_frame: str
    valid: Optional[np.ndarray] = None

    def __post_init__(self):
        coords = frozen_array(self.coordinates, dtype=float)
        confidence = frozen_array(self.confidence, dtype=float)
        parts = tuple(self.bodyparts)
        if coords.ndim != 3 or coords.shape[0] != len(self.time) or coords.shape[2] not in (2, 3):
            raise ValueError("pose coordinates must have shape (time, bodyparts, 2 or 3)")
        if len(parts) != coords.shape[1] or len(set(parts)) != len(parts):
            raise ValueError("bodypart labels must uniquely match the coordinate columns")
        if confidence.shape != coords.shape[:2]:
            raise ValueError("confidence must have shape (time, bodyparts)")
        if np.any(np.isfinite(confidence) & ((confidence < 0) | (confidence > 1))):
            raise ValueError("finite confidence values must be between zero and one")
        valid = frozen_array(
            np.all(np.isfinite(coords), axis=-1) & np.isfinite(confidence)
            if self.valid is None else self.valid, dtype=bool
        )
        if valid.shape != coords.shape[:2] or np.any(
            valid & (~np.all(np.isfinite(coords), axis=-1) | ~np.isfinite(confidence))
        ):
            raise ValueError("pose validity must match bodyparts and cannot mark nonfinite samples valid")
        if not self.unit or not self.coordinate_frame:
            raise ValueError("spatial units and coordinate frame are required")
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "bodyparts", parts)
        object.__setattr__(self, "valid", valid)

__all__ = ["PoseSeries","TOP_USE_BODYPARTS","BOT_RAW_BODYPARTS","BOT_RENAME_MAP","BOT_USE_BODYPARTS","DORSAL_17KP_NAMES","VENTRAL_4KP_NAMES","FUSED_21KP_NAMES","TANDEM_BODYPARTS","TANDEM_23KP","FUSED_23KP_NAMES","BODYPART_REMAP","CHAMBER_MAP","DEFAULT_ARENA_WIDTH","DEFAULT_TARGET_FPS"]
