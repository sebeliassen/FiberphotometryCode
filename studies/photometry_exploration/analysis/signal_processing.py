"""Exploratory compatibility APIs backed by the shared library."""

from fpcode.photometry.aggregation import group_signals as _group_signals
from fpcode.statistics import compute_mean_ci as _compute_mean_ci


def group_signals(sessions, key_fn):
    return _group_signals(sessions, key_fn, missing="raise")


def compute_mean_ci(mat, alpha=0.95):
    return _compute_mean_ci(mat, alpha, empty_policy="numpy")


__all__ = ["group_signals", "compute_mean_ci"]
