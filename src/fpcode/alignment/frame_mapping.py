"""Canonical mapping from model sample coordinates to original video frames."""

import numpy as np


def place_model_labels(labels, original_frames, target_frames):
    """Place integer labels at actual frames; gaps remain invalid and unfilled."""
    labels = np.asarray(labels)
    source = np.asarray(original_frames)
    target = np.asarray(target_frames)
    for name, values in (("labels", labels), ("original frames", source), ("target frames", target)):
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
    if len(labels) != len(source):
        raise ValueError("model labels and original frame indexes must have equal lengths")
    for values in (source, target):
        if np.any(values < 0) or np.any(values[1:] <= values[:-1]):
            raise ValueError("frame indexes must be nonnegative and strictly increasing")
    out = np.full(len(target), -1, dtype=np.int64)
    valid = np.zeros(len(target), dtype=bool)
    if len(source):
        positions = np.searchsorted(source, target)
        bounded = np.minimum(positions, len(source) - 1)
        valid = (positions < len(source)) & (source[bounded] == target) & (labels[bounded] >= 0)
        out[valid] = labels[bounded[valid]]
    return out, valid


def remap_model_frames(table, video_frame_indexes, *, frame_column="frame_index"):
    """Map integer model indices through VFI; return a copy in original row order.

    Unlike historical row-order remapping, the values in frame_column are used
    as model sample indices. Duplicated or unordered video coordinates are
    rejected instead of silently collapsing frames.
    """
    used = np.asarray(video_frame_indexes)
    if used.ndim != 1 or not np.issubdtype(used.dtype, np.integer):
        raise ValueError("video frame indexes must be a one-dimensional integer array")
    if np.any(used < 0) or np.any(used[1:] <= used[:-1]):
        raise ValueError("video frame indexes must be nonnegative and strictly increasing")
    if frame_column not in table:
        raise KeyError(frame_column)
    indices = table[frame_column].to_numpy()
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("model frame coordinates must be integers")
    if np.any(indices < 0) or np.any(indices >= len(used)):
        raise ValueError("model frame coordinate lies outside the VFI mapping")
    out = table.copy()
    out[frame_column] = used[indices]
    return out
