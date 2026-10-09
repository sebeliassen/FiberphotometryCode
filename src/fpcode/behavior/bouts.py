"""Syllable run segmentation with typed, half-open frame bounds."""

from typing import Dict, List, Sequence
import numpy as np
from fpcode.contracts import Bout

def extract_bouts(syllable_seq: Sequence[int], min_length: int = 1) -> List[Bout]:
    """
    Computes run-length encoding of syllable bouts with onset, offset, duration, and syllable ID.
    Bout offsets are half-open [onset, offset).
    """
    seq = np.asarray(syllable_seq, dtype=int)
    if len(seq) == 0:
        return []

    boundaries = np.flatnonzero(np.r_[True, seq[1:] != seq[:-1], True])
    starts = boundaries[:-1]
    ends = boundaries[1:]
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



__all__ = ["extract_bouts", "extract_syllable_runs"]
