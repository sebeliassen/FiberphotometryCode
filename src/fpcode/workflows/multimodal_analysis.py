"""Join modalities through explicit time and original-frame mappings."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fpcode.alignment.frame_mapping import place_model_labels
from fpcode.contracts import PoseSeries, SignalSeries, TimeAxis
from fpcode.contracts.sessions import SessionKey
from fpcode.contracts.timelines import frozen_array
from .photometry import align_photometry


@dataclass(frozen=True)
class MultimodalSession:
    session: SessionKey
    time: TimeAxis
    signals: SignalSeries
    labels: np.ndarray
    label_valid: np.ndarray
    pose: Optional[PoseSeries] = None

    def __post_init__(self):
        labels = frozen_array(self.labels)
        valid = frozen_array(self.label_valid, dtype=bool)
        if labels.shape != (len(self.time),) or valid.shape != labels.shape:
            raise ValueError("labels and validity must match the target timeline")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "label_valid", valid)


def assemble_multimodal_session(session, signals, target, *, model_labels, original_frames,
                                target_frames, pose=None, clock_transform=None,
                                method="exact", tolerance=None, max_gap=None):
    """Align signals and place model labels only at their actual video frames.

    Missing labels stay invalid; trimming gaps are never filled with synthetic
    bouts. Pose must already be on the target timeline, making pose alignment
    and confidence policies explicit upstream.
    """
    target_frames = np.asarray(target_frames)
    if target_frames.shape != (len(target),):
        raise ValueError("target frame indexes must match the target time axis")
    if pose is not None and (pose.time.unit != target.unit or pose.time.clock != target.clock
                             or not np.array_equal(pose.time.values, target.values)):
        raise ValueError("pose must already use the target clock and timeline")
    labels, valid = place_model_labels(model_labels, original_frames, target_frames)
    aligned = align_photometry(signals, target, clock_transform=clock_transform,
                               method=method, tolerance=tolerance, max_gap=max_gap)
    return MultimodalSession(session, target, aligned, labels, valid, pose)
