"""
fpcode.tracking
===============
Synchronized multi-camera posture tracking, clock synchronization,
camera registration, and keypoint fusion engines.
"""

from importlib import import_module

# Geometry and confidence primitives are usable without loading cohort
# discovery, calibration JSON or pandas/HDF fusion implementations.
_EXPORTS = {
    "find_matched_pairs": "pair_matching",
    "extract_timestamp_from_filename": "pair_matching",
    "normalize_arena_name": "pair_matching",
    "load_ttl_cache": "clock_sync",
    "find_camera_timestamp_files": "clock_sync",
    "compute_pair_ttl_offset": "clock_sync",
    "build_and_save_ttl_cache": "clock_sync",
    "get_session_ttl_offset": "clock_sync",
    "top_to_bot_frame": "clock_sync",
    "bot_to_top_frame": "clock_sync",
    "compute_session_camera_mapping": "camera_correspondence",
    "CameraMappingContext": "camera_correspondence",
    "get_session_mapping": "camera_correspondence",
    "get_session_calibration_details": "camera_correspondence",
    "top_to_bot_frame_continuous": "camera_correspondence",
    "fuse_multiview_session": "fusion",
    "fuse_multiview_static_21kp": "static_fusion",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
