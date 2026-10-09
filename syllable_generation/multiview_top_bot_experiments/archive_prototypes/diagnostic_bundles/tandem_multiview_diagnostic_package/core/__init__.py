"""
core
====
Shared foundation modules for multiview (TopCam + BotCam) Keypoint-MoSeq pipeline:
- schemas: Canonical keypoint definitions, body parts, and hyperparameters
- logging_utils: Structured logging and notifications
- pair_matching: Cross-camera video/tracking alignment and metadata matching
- fusion_engine: 4 Golden Rules mathematical fusion engine
- io_utils: Safe HDF5, checkpoint, and pickle serialization
- evaluation_metrics: EML computation, syllable dynamics, and transition analysis
"""

from .fusion_engine import fuse_single_pair, rot2d, smooth_centroid_median, smooth_heading_circular
from .logging_utils import get_logger, notify_status
from .pair_matching import find_matched_pairs
from .schemas import (
    BODYPART_REMAP,
    BOT_USE_BODYPARTS,
    CHAMBER_MAP,
    TANDEM_BODYPARTS,
    TOP_USE_BODYPARTS,
    PipelineHyperparameters,
)

__all__ = [
    "TOP_USE_BODYPARTS",
    "BOT_USE_BODYPARTS",
    "TANDEM_BODYPARTS",
    "BODYPART_REMAP",
    "CHAMBER_MAP",
    "PipelineHyperparameters",
    "get_logger",
    "notify_status",
    "find_matched_pairs",
    "rot2d",
    "fuse_single_pair",
    "smooth_heading_circular",
    "smooth_centroid_median",
]
