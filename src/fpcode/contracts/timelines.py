"""Canonical coordinates for sampled data with explicit units and clock identity."""

from dataclasses import dataclass

import numpy as np


def frozen_array(value, *, dtype=None):
    """Own a read-only copy so callers cannot mutate a contract through an input."""
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _exact_integer_coordinates(values):
    """Return exact whole-number coordinates as integers when representable."""
    if np.issubdtype(values.dtype, np.integer):
        return values
    if not np.issubdtype(values.dtype, np.floating) or not np.all(values == np.floor(values)):
        return None
    if not len(values):
        return values.astype(np.int64)
    minimum, maximum = int(values[0]), int(values[-1])
    if np.iinfo(np.int64).min <= minimum and maximum <= np.iinfo(np.int64).max:
        return values.astype(np.int64)
    if 0 <= minimum and maximum <= np.iinfo(np.uint64).max:
        return values.astype(np.uint64)
    return None


def _relative_integer_coordinates(*arrays):
    """Rebase integer axes together without signed/unsigned float promotion."""
    nonempty = [array for array in arrays if len(array)]
    origin = min((int(array[0]) for array in nonempty), default=0)
    extent = max((int(array[-1]) for array in nonempty), default=origin) - origin
    if extent <= np.iinfo(np.uint64).max:
        # Modular casting/subtraction also handles negative signed originals;
        # the resulting offsets are nonnegative and fit the checked extent.
        modulo_origin = np.uint64(origin % (1 << 64))
        relative = tuple(array.astype(np.uint64) - modulo_origin for array in arrays)
    else:
        # Rare coordinates spanning negative int64 and the full uint64 range.
        relative = tuple(array.astype(object) - origin for array in arrays)
    return relative, origin


def _shift_coordinates(values, offset, *, scale=1):
    """Apply an integral scale/origin exactly, including whole float inputs."""
    integer_values = _exact_integer_coordinates(values)
    if integer_values is None or offset != int(offset) or scale != int(scale):
        return values * scale + offset
    values = integer_values
    if not len(values):
        return values.copy()
    offset = int(offset)
    scale = int(scale)
    minimum, maximum = int(values[0]) * scale + offset, int(values[-1]) * scale + offset
    if np.iinfo(np.int64).min <= minimum and maximum <= np.iinfo(np.int64).max:
        dtype = np.int64
    elif 0 <= minimum and maximum <= np.iinfo(np.uint64).max:
        dtype = np.uint64
    else:
        raise ValueError("shifted integer coordinates exceed the supported 64-bit range")
    shifted = values.astype(np.uint64) * np.uint64(scale % (1 << 64)) + np.uint64(offset % (1 << 64))
    return shifted.astype(dtype)


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
