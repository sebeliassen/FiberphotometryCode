"""
core.evaluation_metrics
=======================
Evaluation routines for Keypoint-MoSeq models:
- Fast syllable frequency sorting & reindexing
- Results extraction (syllable sequences, latent states)
- Syllable vocabulary & duration distribution analysis
- Bigram transition probability matrix computation
- Expected Marginal Likelihood (EML) calculation
"""

from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np

from ..common.logging import get_logger

logger = get_logger("fpcode.kpms.metrics")


def get_frequencies(z: np.ndarray, mask: np.ndarray, num_states: int, runlength: bool = True) -> np.ndarray:
    """Computes syllable usage frequencies (either frame-count or run-length/occurrence)."""
    counts = np.zeros(num_states, dtype=np.int64)
    if not runlength:
        valid_z = z[mask]
        for s in range(num_states):
            counts[s] = np.sum(valid_z == s)
        return counts

    # Run-length frequency
    for row_z, row_mask in zip(z, mask):
        valid_row = row_z[row_mask]
        if len(valid_row) == 0:
            continue
        changes = np.diff(valid_row) != 0
        starts = np.insert(changes, 0, True)
        run_sylls = valid_row[starts]
        for s in run_sylls:
            if s < num_states:
                counts[s] += 1
    return counts


def extract_syllable_statistics(results_h5_path: Path) -> Dict[str, Any]:
    """
    Extracts syllable vocabulary metrics, active syllable count,
    usage frequencies, duration distributions, and bigram transition matrix.
    """
    path = Path(results_h5_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with h5py.File(path, "r") as f:
        # Check standard results structures
        syllables = {}
        if "syllables" in f:
            for k in f["syllables"].keys():
                syllables[k] = f["syllables"][k][()]
        elif "states/z" in f:
            for k in f["states/z"].keys():
                syllables[k] = f["states/z"][k][()]
        else:
            # Check for per-session groups (e.g. /<session_id>/syllable)
            for k in f.keys():
                if isinstance(f[k], h5py.Group) and "syllable" in f[k]:
                    syllables[k] = f[k]["syllable"][()]

        if not syllables:
            raise KeyError(f"No 'syllable' datasets found in {path}")

    all_durations: List[int] = []
    flat_sylls: List[int] = []
    all_runs: List[int] = []

    for seq in syllables.values():
        if len(seq) == 0:
            continue
        flat_sylls.extend(seq)
        changes = np.diff(seq) != 0
        run_lengths = np.diff(np.where(np.insert(np.append(changes, True), 0, True))[0])
        run_labels = seq[np.insert(changes, 0, True)]
        all_durations.extend(run_lengths)
        all_runs.extend(run_labels)

    flat_arr = np.array(flat_sylls, dtype=np.int32)
    run_arr = np.array(all_runs, dtype=np.int32)
    dur_arr = np.array(all_durations, dtype=np.int32)

    unique_sylls, frame_counts = np.unique(flat_arr, return_counts=True)
    _, run_counts = np.unique(run_arr, return_counts=True)

    max_state = int(np.max(unique_sylls)) + 1
    # Compute bigram transition matrix
    T = np.zeros((max_state, max_state), dtype=np.float64)
    for seq in syllables.values():
        if len(seq) < 2:
            continue
        changes = np.diff(seq) != 0
        run_labels = seq[np.insert(changes, 0, True)]
        for i in range(len(run_labels) - 1):
            s_from = run_labels[i]
            s_to = run_labels[i + 1]
            if s_from < max_state and s_to < max_state:
                T[s_from, s_to] += 1.0

    # Row normalize transitions
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    T_norm = T / row_sums

    return {
        "num_sessions": len(syllables),
        "total_frames": int(len(flat_arr)),
        "active_syllables": int(len(unique_sylls)),
        "unique_syllables": unique_sylls.tolist(),
        "frame_counts": frame_counts.tolist(),
        "run_counts": run_counts.tolist(),
        "median_duration_frames": float(np.median(dur_arr)) if len(dur_arr) > 0 else 0.0,
        "mean_duration_frames": float(np.mean(dur_arr)) if len(dur_arr) > 0 else 0.0,
        "transition_matrix": T_norm,
    }
