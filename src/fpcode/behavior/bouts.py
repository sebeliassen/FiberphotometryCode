"""Syllable run segmentation with typed, half-open frame bounds."""

from typing import Dict, List, Sequence

import numpy as np

from fpcode.contracts import Bout


def run_boundaries(values, *, frame_indices=None):
    """Return half-open row bounds for equal adjacent values.

    Missing values remain in the stream: callers may replace them with a gap
    sentinel, but must not remove them before finding transitions. By default,
    adjacency means neighboring rows. Supplying increasing integer original
    frame indices also splits runs at dropped frames; bounds still index rows.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("run values must be one-dimensional")
    contiguous = _contiguous_rows(len(values), frame_indices)
    if not len(values):
        return np.array([], dtype=int), np.array([], dtype=int)
    changes = (values[1:] != values[:-1]) | ~contiguous
    boundaries = np.flatnonzero(np.r_[True, changes, True])
    return boundaries[:-1], boundaries[1:]


def _contiguous_rows(length, frame_indices):
    if frame_indices is None:
        return np.ones(max(0, length - 1), dtype=bool)
    frames = np.asarray(frame_indices)
    if frames.ndim != 1 or len(frames) != length or frames.dtype.kind not in "iu":
        raise ValueError("original frame indices must be matching one-dimensional integers")
    if np.any(frames[1:] <= frames[:-1]):
        raise ValueError("original frame indices must increase strictly")
    # A preceding frame cannot be the dtype maximum if a larger next frame
    # exists, so this addition cannot wrap at the integer range boundary.
    return frames[1:] == frames[:-1] + 1


def extract_bouts(syllable_seq: Sequence[int], min_length: int = 1) -> List[Bout]:
    """
    Computes run-length encoding of syllable bouts with onset, offset, duration, and syllable ID.
    Bout offsets are half-open [onset, offset).
    """
    seq = np.asarray(syllable_seq, dtype=int)
    if len(seq) == 0:
        return []

    starts, ends = run_boundaries(seq)
    sylls = seq[starts]

    bouts = []
    for s, e, syll in zip(starts, ends, sylls):
        dur = int(e - s)
        if dur >= min_length:
            bouts.append(Bout(int(syll), int(s), int(e)))
    return bouts


def extract_syllable_runs(syllable_seq: Sequence[int], min_length: int = 1) -> List[Dict[str, int]]:
    """Historical dictionary representation of typed half-open bouts."""
    return [dict(syllable=b.syllable, onset=b.onset, offset=b.offset, duration=b.duration)
            for b in extract_bouts(syllable_seq, min_length)]



__all__ = ["run_boundaries", "extract_bouts", "extract_syllable_runs"]
