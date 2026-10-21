"""Strict coordinate-aware alignment; no endpoint padding or implicit gap repair."""

import numpy as np

from fpcode.contracts import SignalSeries, TimeAxis
from fpcode.contracts.timelines import _exact_integer_coordinates, _relative_integer_coordinates


def align_signal(source: SignalSeries, target: TimeAxis, *, method="exact", tolerance=None, max_gap=None):
    """Resample signals without changing their clock, units or validity meaning.

    Exact matches are the default. Nearest matching requires a tolerance and
    chooses the earlier coordinate on ties. Linear interpolation requires an
    explicit maximum coordinate gap and two valid adjacent samples. All modes
    leave samples outside the source extent invalid; no endpoint extrapolation.
    """
    if source.time.unit != target.unit or source.time.clock != target.clock:
        raise ValueError("alignment requires matching time units and clock identity")
    if method not in {"exact", "nearest", "linear"}:
        raise ValueError("method must be exact, nearest, or linear")
    if method == "nearest" and (tolerance is None or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("nearest matching requires a finite nonnegative tolerance")
    if method == "linear" and (max_gap is None or not np.isfinite(max_gap) or max_gap <= 0):
        raise ValueError("linear interpolation requires a finite positive max_gap")

    x, q = source.time.values, target.values
    integer_x, integer_q = _exact_integer_coordinates(x), _exact_integer_coordinates(q)
    if integer_x is not None and integer_q is not None:
        (x, q), _ = _relative_integer_coordinates(integer_x, integer_q)
    scalar = source.values.ndim == 1
    values = source.values[:, None] if scalar else source.values
    valid = source.valid[:, None] if scalar else source.valid
    out = np.full((len(q), values.shape[1]), np.nan)
    mask = np.zeros(out.shape, dtype=bool)
    if len(x):
        positions = np.searchsorted(x, q)
        right = np.clip(positions, 0, len(x) - 1)
        left = np.clip(positions - 1, 0, len(x) - 1)
        inside = (q >= x[0]) & (q <= x[-1])
        exact = inside & (x[right] == q)

        chosen = right.copy()
        matched = exact.copy()
        if method == "nearest":
            # Nonnegative ordered differences work for unsigned coordinates,
            # including targets outside the source extent that stay invalid.
            left_distance = np.maximum(q, x[left]) - np.minimum(q, x[left])
            right_distance = np.maximum(q, x[right]) - np.minimum(q, x[right])
            chosen = np.where(left_distance <= right_distance, left, right)
            matched = inside & (np.minimum(left_distance, right_distance) <= tolerance)
        out[matched] = values[chosen[matched]]
        mask[matched] = valid[chosen[matched]]

        if method == "linear":
            between = inside & ~exact & (positions > 0) & (positions < len(x))
            rows = np.flatnonzero(between)
            gaps = x[right[rows]] - x[left[rows]]
            rows = rows[gaps <= max_gap]
            if len(rows):
                lo, hi = left[rows], right[rows]
                weights = np.asarray((q[rows] - x[lo]) / (x[hi] - x[lo]), dtype=float)[:, None]
                out[rows] = values[lo] + weights * (values[hi] - values[lo])
                mask[rows] = valid[lo] & valid[hi]

    out[~mask] = np.nan
    return SignalSeries(target, out[:, 0] if scalar else out, source.signals, mask[:, 0] if scalar else mask)
