"""Explicit confidence thresholds applied to immutable labelled pose data."""

import numpy as np

from fpcode.contracts import PoseSeries


def confidence_mask(pose: PoseSeries, *, threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("confidence threshold must be between zero and one")
    return pose.valid & (pose.confidence >= threshold)


def gate_pose(pose: PoseSeries, *, threshold):
    mask = confidence_mask(pose, threshold=threshold)
    return PoseSeries(pose.time, pose.coordinates, pose.confidence, pose.bodyparts,
                      pose.unit, pose.coordinate_frame, mask)
