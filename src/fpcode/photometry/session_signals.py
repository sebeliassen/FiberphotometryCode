"""Shared preparation for historical monoamine sessions, with named policies."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from fpcode.adapters.legacy_sessions import photometry_table, region_columns

from .event_windows import legacy_nearest_window
from .preprocessing import motion_correct_biexponential


@dataclass
class PreparedSignals:
    timestamps: np.ndarray
    traces: Dict[str, np.ndarray]
    failures: Dict[str, str]


def prepare_region_traces(session, *, normalization="none", errors="raise", corrector=None):
    """Correct each region once, preserving first-column lookup and sample order.

    normalization='global_zscore' uses the notebook's historical NaN-excluding
    mean/std convention. errors='skip' records failures rather than hiding them.
    No interpolation, finite-row removal, epoch buffer or alignment is performed.
    """
    if normalization not in {"none", "global_zscore"} or errors not in {"raise", "skip"}:
        raise ValueError("unknown normalization or error policy")
    corrector = motion_correct_biexponential if corrector is None else corrector
    raw470, raw415 = photometry_table(session, "470"), photometry_table(session, "415")
    mapping = region_columns(session)
    traces, failures = {}, {}
    for region in mapping.values():
        if region in traces or region in failures:
            continue
        column = next(key for key, value in mapping.items() if value == region)
        try:
            trace = np.array(corrector(raw470[column].values, raw415[column].values), copy=True)
            if normalization == "global_zscore":
                valid = ~np.isnan(trace)
                normalized = np.empty_like(trace)
                normalized[:] = np.nan
                normalized[valid] = (trace[valid] - np.nanmean(trace[valid])) / np.nanstd(trace[valid])
                trace = normalized
            traces[region] = trace
        except Exception as exc:
            if errors == "raise":
                raise
            failures[region] = str(exc)
    return PreparedSignals(raw470["SystemTimestamp"].to_numpy(copy=True), traces, failures)


__all__ = ["PreparedSignals", "prepare_region_traces", "legacy_nearest_window"]
