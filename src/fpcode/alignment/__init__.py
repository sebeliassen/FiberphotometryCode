"""Coordinate-based numerical alignment, independent of acquisition formats."""

from .frame_mapping import remap_model_frames
from .resampling import align_signal

__all__ = ["align_signal", "remap_model_frames"]
