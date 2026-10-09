"""
fpcode.tracking
===============
Synchronized multi-camera posture tracking, clock synchronization,
camera registration, and keypoint fusion engines.
"""

from .camera_correspondence import (
    compute_session_camera_mapping,
    get_session_calibration_details,
    get_session_mapping,
    top_to_bot_frame_continuous,
)
from .clock_sync import (
    bot_to_top_frame,
    build_and_save_ttl_cache,
    compute_pair_ttl_offset,
    find_camera_timestamp_files,
    get_session_ttl_offset,
    load_ttl_cache,
    top_to_bot_frame,
)
from .fusion import fuse_multiview_session
from .pair_matching import (
    extract_timestamp_from_filename,
    find_matched_pairs,
    normalize_arena_name,
)
from .static_fusion import fuse_multiview_static_21kp

__all__ = [
    "find_matched_pairs",
    "extract_timestamp_from_filename",
    "normalize_arena_name",
    "load_ttl_cache",
    "find_camera_timestamp_files",
    "compute_pair_ttl_offset",
    "build_and_save_ttl_cache",
    "get_session_ttl_offset",
    "top_to_bot_frame",
    "bot_to_top_frame",
    "compute_session_camera_mapping",
    "get_session_mapping",
    "get_session_calibration_details",
    "top_to_bot_frame_continuous",
    "fuse_multiview_session",
    "fuse_multiview_static_21kp",
]
