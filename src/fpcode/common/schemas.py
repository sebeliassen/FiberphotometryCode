"""Compatibility facade for historical schemas and machine-specific PathConfig.

New numerical code imports keypoint layouts from fpcode.contracts.pose. Existing
PathConfig defaults remain unchanged; new workflows accept paths explicitly.
"""

from dataclasses import dataclass, field
from pathlib import Path

from fpcode.contracts.model import PipelineHyperparameters
from fpcode.contracts.pose import (
    TOP_USE_BODYPARTS,
    BOT_RAW_BODYPARTS,
    BOT_RENAME_MAP,
    BOT_USE_BODYPARTS,
    DORSAL_17KP_NAMES,
    VENTRAL_4KP_NAMES,
    FUSED_21KP_NAMES,
    TANDEM_BODYPARTS,
    TANDEM_23KP,
    FUSED_23KP_NAMES,
    BODYPART_REMAP,
    CHAMBER_MAP,
    DEFAULT_ARENA_WIDTH,
    DEFAULT_TARGET_FPS,
)

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

__all__ = ["TOP_USE_BODYPARTS","BOT_RAW_BODYPARTS","BOT_RENAME_MAP","BOT_USE_BODYPARTS","DORSAL_17KP_NAMES","VENTRAL_4KP_NAMES","FUSED_21KP_NAMES","TANDEM_BODYPARTS","TANDEM_23KP","FUSED_23KP_NAMES","BODYPART_REMAP","CHAMBER_MAP","DEFAULT_ARENA_WIDTH","DEFAULT_TARGET_FPS","PipelineHyperparameters","PathConfig"]
