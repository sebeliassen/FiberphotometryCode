"""
fpcode.behavior.syllables
=========================
Syllable bout segmentation, run length extraction, and timeline alignment.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def extract_syllable_runs(syllable_seq: Sequence[int], min_length: int = 1) -> List[Dict[str, int]]:
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
            bouts.append(
                {
                    "syllable": int(syll),
                    "onset": int(s),
                    "offset": int(e),
                    "duration": dur,
                }
            )
    return bouts


def tile_results(results: Dict[str, dict], factor: int) -> None:
    """
    Tiles all 1D/2D numpy arrays in the results dictionary by an integer factor in place.
    """
    if factor <= 1:
        return
    for k in results:
        for key, val in results[k].items():
            if isinstance(val, (np.ndarray, list)):
                a = np.asarray(val)
                if a.ndim == 1:
                    results[k][key] = np.repeat(a, factor)
                else:
                    results[k][key] = np.repeat(a, factor, axis=0)


def flatten_results_to_timeline(
    results: Dict[str, dict],
    vfi: Dict[str, np.ndarray],
    required_keys: Sequence[str] = ("syllable", "latent_state", "centroid", "heading"),
    actual_lengths: Optional[Dict[str, int]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Maps Keypoint-MoSeq results onto absolute video timeline using Video Frame Indexes (VFI).
    """
    flattened = {}
    for k, vdict in results.items():
        if k not in vfi:
            continue
        idx = np.asarray(vfi[k], dtype=int)
        full_len = actual_lengths.get(k) if actual_lengths else (int(np.max(idx)) + 1 if len(idx) > 0 else 0)

        df = pd.DataFrame(index=np.arange(full_len))
        for key in required_keys:
            if key not in vdict:
                continue
            val = np.asarray(vdict[key])
            if val.ndim == 1:
                col_series = pd.Series(val, index=idx[: len(val)])
                df[key] = col_series
            elif val.ndim == 2:
                for c in range(val.shape[1]):
                    col_series = pd.Series(val[:, c], index=idx[: len(val)])
                    df[f"{key}_{c}"] = col_series
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        flattened[k] = df
    return flattened
