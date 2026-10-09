"""
core.schemas
============
Single Source of Truth (SSOT) for:
- Keypoint layout & bodypart names (dorsal, ventral, tandem 23-kp)
- Camera chamber coordinate mapping
- Training & inference hyperparameters
- Standard file naming and path schemas
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# -----------------------------------------------------------------------------
# 1. Body Part Layouts & Schemas
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 3. Model Training & Convergence Hyperparameters
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineHyperparameters:
    num_ar_iters: int = 50
    num_slds_iters: int = 200
    total_iters: int = 250
    save_every_n_iters: int = 25
    init_kappa: float = 7.5e13
    decrease_kappa_factor: float = 10.0
    slds_kappa: float = 7.5e12
    target_fps: int = 60
    conf_threshold: float = 0.5
    min_frames: int = 100


@dataclass
class PathConfig:
    repo_root: Path = Path("/home/fsp585/Desktop/FiberphotometryCode")
    experiment_dir: Path = field(
        default_factory=lambda: Path(
            "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments"
        )
    )
    top_h5_dir: Path = field(
        default_factory=lambda: Path("/home/fsp585/Desktop/FiberphotometryCode/karen_project/h5_files")
    )
    bot_prod_dir: Path = field(
        default_factory=lambda: Path("/home/fsp585/Desktop/production_analysis_arm_B_seed_123_20260922/predictions")
    )
    fused_dir: Path = field(
        default_factory=lambda: Path(
            "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/fused_h5_full_cohort"
        )
    )
    prod_kpms_dir: Path = field(
        default_factory=lambda: Path(
            "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/kpms_tandem_cohort_production"
        )
    )
