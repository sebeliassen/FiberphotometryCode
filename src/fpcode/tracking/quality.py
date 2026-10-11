"""Explicit confidence thresholds applied to immutable labelled pose data."""

import numpy as np

from fpcode.contracts import PoseSeries


def supported_track_mask(coordinates, confidence, *, threshold):
    """Raw 2D support from finite x/y and finite thresholded confidence.

    Threshold selection belongs to the caller's policy. This array interface
    retains historical visualization support rules and does not fill gaps.
    """
    coordinates = np.asarray(coordinates)
    confidence = np.asarray(confidence)
    finite_xy = np.isfinite(coordinates[..., 0]) & np.isfinite(coordinates[..., 1])
    with np.errstate(invalid="ignore"):
        confident = np.isfinite(confidence) & (confidence >= threshold)
    return finite_xy & confident


def confidence_mask(pose: PoseSeries, *, threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("confidence threshold must be between zero and one")
    return pose.valid & supported_track_mask(pose.coordinates, pose.confidence, threshold=threshold)


def gate_pose(pose: PoseSeries, *, threshold):
    mask = confidence_mask(pose, threshold=threshold)
    return PoseSeries(pose.time, pose.coordinates, pose.confidence, pose.bodyparts,
                      pose.unit, pose.coordinate_frame, mask)
