"""Canonical coordinates for sampled data with explicit units and clock identity."""

from dataclasses import dataclass

import numpy as np


def frozen_array(value, *, dtype=None):
    """Own a read-only copy so callers cannot mutate a contract through an input."""
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class TimeAxis:
    values: np.ndarray
    unit: str
    clock: str

    def __post_init__(self):
        values = frozen_array(self.values)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
            raise ValueError("time coordinates must be a one-dimensional numeric array")
        if not np.all(np.isfinite(values)) or np.any(values[1:] <= values[:-1]):
            raise ValueError("time coordinates must be finite and strictly increasing")
        if self.unit not in {"s", "ms", "ns", "frame"}:
            raise ValueError("time unit must be s, ms, ns, or frame")
        if self.unit == "frame" and np.any(values != np.floor(values)):
            raise ValueError("frame coordinates must be integers")
        if not isinstance(self.clock, str) or not self.clock.strip():
            raise ValueError("clock identity is required")
        object.__setattr__(self, "values", values)

    def __len__(self):
        return len(self.values)
