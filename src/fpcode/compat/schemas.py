"""Compatibility facade for historical schemas and machine-specific PathConfig.

New numerical code imports keypoint layouts from fpcode.contracts.pose. Historical
PathConfig defaults resolve from configured roots; new workflows accept paths explicitly.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from fpcode.common.paths import get_repo_root, resolve_data_path
from fpcode.contracts.model import PipelineHyperparameters
from fpcode.contracts.pose import (
    BODYPART_REMAP,
    BOT_RAW_BODYPARTS,
    BOT_RENAME_MAP,
    BOT_USE_BODYPARTS,
    CHAMBER_MAP,
    DEFAULT_ARENA_WIDTH,
    DEFAULT_TARGET_FPS,
    DORSAL_17KP_NAMES,
    FUSED_21KP_NAMES,
    FUSED_23KP_NAMES,
    TANDEM_23KP,
    TANDEM_BODYPARTS,
    TOP_USE_BODYPARTS,
    VENTRAL_4KP_NAMES,
)


def _bottom_tracking_root():
    override = os.environ.get("FP_BOT_TRACKING_ROOT", "").strip()
    return Path(override).expanduser().resolve() if override else resolve_data_path(
        "data/raw/karen_pff_vs_monomer/bot_tracking_predictions"
    )


@dataclass
class PathConfig:
    """Historical multiview paths resolved from explicit environment roots."""

    repo_root: Path = field(default_factory=get_repo_root)
    experiment_dir: Path = field(default_factory=lambda: get_repo_root() / "studies/multiview_experiments")
    top_h5_dir: Path = field(default_factory=lambda: resolve_data_path("data/raw/karen_pff_vs_monomer/h5_files"))
    bot_prod_dir: Path = field(default_factory=_bottom_tracking_root)
    fused_dir: Path = field(default_factory=lambda: resolve_data_path("data/fused/multiview/fused_h5_full_cohort"))
    prod_kpms_dir: Path = field(default_factory=lambda: resolve_data_path("data/models/kpms_tandem_cohort_production"))


# Retain the historical qualified class name for newly written and old pickles.
PathConfig.__module__ = "fpcode.common.schemas"

__all__ = ["TOP_USE_BODYPARTS","BOT_RAW_BODYPARTS","BOT_RENAME_MAP","BOT_USE_BODYPARTS","DORSAL_17KP_NAMES","VENTRAL_4KP_NAMES","FUSED_21KP_NAMES","TANDEM_BODYPARTS","TANDEM_23KP","FUSED_23KP_NAMES","BODYPART_REMAP","CHAMBER_MAP","DEFAULT_ARENA_WIDTH","DEFAULT_TARGET_FPS","PipelineHyperparameters","PathConfig"]
