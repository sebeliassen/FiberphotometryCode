"""Frame and bout counts with explicit vocabulary and missing-frame boundaries."""

import numpy as np

from fpcode.behavior.bouts import _contiguous_rows, run_boundaries


def label_counts(labels, num_states, *, mode="frames", frame_indices=None):
    """Count valid integer labels without joining runs across missing data.

    Negative and out-of-vocabulary labels remain boundaries but are not counted.
    Study-specific coercion, weighting and duration denominators belong to callers.
    Optional original frame indexes split bouts across dropped frames.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("labels must be a one-dimensional integer array")
    if isinstance(num_states, bool) or not isinstance(num_states, (int, np.integer)) or num_states < 1:
        raise ValueError("num_states must be a positive integer")
    if mode not in {"frames", "bouts"}:
        raise ValueError("mode must be frames or bouts")
    if mode == "bouts":
        starts, _ = run_boundaries(labels, frame_indices=frame_indices)
        labels = labels[starts]
    else:
        _contiguous_rows(len(labels), frame_indices)
    valid = labels[(labels >= 0) & (labels < num_states)]
    return np.bincount(valid.astype(np.int64), minlength=int(num_states))
