"""Frame-resolved body geometry and projected paw movement with confidence masks."""

import numpy as np


def body_relative_features(pose, confidence, *, landmarks, confidence_threshold, frames, fps):
    """Measure 2D posture and paws; never infer contacts, height, or millimetres.

    Explicit landmark indices identify nose, tail, neck, left/right shoulders,
    left/right hips and four paws. Paw coordinates are projected along/across
    the nose-tail axis and divided by instantaneous projected body length.
    """
    pose, confidence = np.asarray(pose), np.asarray(confidence)
    if pose.ndim != 3 or pose.shape[2] != 2 or confidence.shape != pose.shape[:2]:
        raise ValueError("pose/confidence dimensions disagree")
    if len(frames) != len(pose) or fps <= 0:
        raise ValueError("explicit frame clock required")
    nose, tail = landmarks["nose"], landmarks["tail"]
    axis = pose[:, nose] - pose[:, tail]
    length = np.linalg.norm(axis, axis=1)
    axial = axis / np.maximum(length[:, None], 1e-9)
    transverse = np.column_stack((-axial[:, 1], axial[:, 0]))
    good = (confidence[:, nose] >= confidence_threshold) & (confidence[:, tail] >= confidence_threshold)
    good &= np.isfinite(axis).all(axis=1) & (length > 1)
    features = {"body_length_px": np.where(good, length, np.nan)}
    for name, left, right in (("shoulder_width_ratio", "left_shoulder", "right_shoulder"),
                              ("hip_width_ratio", "left_hip", "right_hip")):
        li, ri = landmarks[left], landmarks[right]
        supported = good & (confidence[:, li] >= confidence_threshold) & (confidence[:, ri] >= confidence_threshold)
        value = np.linalg.norm(pose[:, li] - pose[:, ri], axis=1) / np.maximum(length, 1e-9)
        features[name] = np.where(supported, value, np.nan)
    neck = landmarks["neck"]
    head = pose[:, nose] - pose[:, neck]
    head_angle = np.arctan2(np.sum(head * transverse, axis=1), np.sum(head * axial, axis=1))
    features["head_axis_angle_rad"] = np.where(good & (confidence[:, neck] >= confidence_threshold), head_angle, np.nan)
    velocities = {}
    for name in ("left_front", "right_front", "left_hind", "right_hind"):
        index = landmarks[name]
        supported = good & (confidence[:, index] >= confidence_threshold)
        displacement = pose[:, index] - pose[:, tail]
        x = np.sum(displacement * axial, axis=1) / np.maximum(length, 1e-9)
        y = np.sum(displacement * transverse, axis=1) / np.maximum(length, 1e-9)
        xy = np.column_stack((x, y))
        velocity = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps]
        adjacent = np.r_[False, supported[1:] & supported[:-1] & (np.diff(frames) == 1)]
        velocity[~adjacent] = np.nan
        features[name + "_axial_ratio"] = np.where(supported, x, np.nan)
        features[name + "_lateral_ratio"] = np.where(supported, y, np.nan)
        features[name + "_speed_bodylengths_s"] = velocity
        velocities[name] = velocity
    for pair in ("front", "hind"):
        left, right = velocities["left_" + pair], velocities["right_" + pair]
        features[pair + "_speed_asymmetry"] = abs(left - right) / np.maximum(left + right, 1e-9)
    return features
