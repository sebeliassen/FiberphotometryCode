"""Canonical normal-interval summaries with explicit historical policies."""

import numpy as np
from scipy import stats


def compute_mean_ci(mat, alpha=0.95, *, empty_policy="zeros"):
    """Mean +/- normal interval using population SD, preserving legacy choices.

    empty_policy='zeros' retains fpcode's existing empty-input result.
    empty_policy='numpy' retains the exploratory study's NumPy/NaN behavior.
    This is a trial-level summary; it does not infer animal-level replication.
    """
    if empty_policy not in {"zeros", "numpy"}:
        raise ValueError("empty_policy must be zeros or numpy")
    mat = np.asarray(mat)
    if empty_policy == "zeros" and (mat.ndim != 2 or mat.shape[0] == 0):
        empty = np.zeros(mat.shape[-1] if mat.ndim else 0)
        return empty, empty, empty
    ys = np.mean(mat, axis=0)
    sem = np.std(mat, axis=0) / np.sqrt(mat.shape[0])
    lo, hi = stats.norm.interval(alpha, loc=ys, scale=sem)
    return ys, lo, hi
