"""Explicit clock calibration and time-unit conversion without fitting policy."""

from dataclasses import dataclass
import math

import numpy as np

from fpcode.contracts import TimeAxis
from fpcode.contracts.timelines import _exact_integer_coordinates, _shift_coordinates

_SECONDS_PER_UNIT = {"s": 1.0, "ms": 1e-3, "ns": 1e-9}


def _subtract_origin(values, origin):
    """Subtract calibration origins before converting integer times to float."""
    integer_values = _exact_integer_coordinates(values)
    if integer_values is None:
        return values - origin
    values = integer_values
    integer_origin = int(origin)
    # Split even a float-valued origin: subtracting it directly from int64
    # would convert the absolute timestamps to float before rebasing. Rebasing
    # also allows negative offsets from unsigned timestamps without wrapping.
    try:
        difference = _shift_coordinates(values, -integer_origin)
    except ValueError:
        # A calibrated difference can exceed both signed and unsigned 64-bit
        # ranges. Python subtraction still gives the correct float rounding.
        difference = values.astype(object) - integer_origin
    return np.asarray(difference, dtype=float) - (origin - integer_origin)


@dataclass(frozen=True)
class ClockTransform:
    source_clock: str
    target_clock: str
    source_unit: str
    target_unit: str
    source_origin: float
    target_origin: float
    scale: float

    def __post_init__(self):
        if not self.source_clock or not self.target_clock:
            raise ValueError("source and target clocks are required")
        if self.source_unit not in {*_SECONDS_PER_UNIT, "frame"} or self.target_unit not in {*_SECONDS_PER_UNIT, "frame"}:
            raise ValueError("unsupported clock coordinate unit")
        if not all(math.isfinite(x) for x in (self.source_origin, self.target_origin, self.scale)) or self.scale <= 0:
            raise ValueError("clock origins must be finite and scale positive")

    def apply(self, axis: TimeAxis) -> TimeAxis:
        if axis.clock != self.source_clock or axis.unit != self.source_unit:
            raise ValueError("source clock/unit does not match the calibration")
        integer_values = _exact_integer_coordinates(axis.values)
        integer_scale = int(self.scale)
        if integer_values is not None and self.scale == integer_scale:
            source_whole, target_whole = int(self.source_origin), int(self.target_origin)
            fractional_offset = ((self.target_origin - target_whole)
                                 - (self.source_origin - source_whole) * integer_scale)
            if fractional_offset == int(fractional_offset):
                offset = target_whole - source_whole * integer_scale + int(fractional_offset)
                return TimeAxis(_shift_coordinates(integer_values, offset, scale=integer_scale),
                                self.target_unit, self.target_clock)
        # Subtract origins before floating conversion, then add whole target
        # origins exactly if mapped coordinates are integral. Fractional
        # offsets at large absolute epochs require a rebased target clock.
        delta = _subtract_origin(axis.values, self.source_origin)
        mapped = np.asarray(delta, dtype=float) * self.scale
        if abs(self.target_origin) >= 2**53 and np.any(mapped != np.floor(mapped)):
            raise ValueError("fractional offsets at a large absolute origin require rebasing the target clock")
        values = _shift_coordinates(mapped, self.target_origin)
        return TimeAxis(values, self.target_unit, self.target_clock)


def convert_time_unit(axis: TimeAxis, unit: str, *, origin=0) -> TimeAxis:
    """Convert physical time units, optionally rebasing at an explicit origin.

    Frames require a calibrated ClockTransform; no FPS or clock is inferred.
    """
    if axis.unit not in _SECONDS_PER_UNIT or unit not in _SECONDS_PER_UNIT:
        raise ValueError("frame conversion requires an explicit clock calibration")
    return ClockTransform(axis.clock, axis.clock, axis.unit, unit, origin, 0,
                          _SECONDS_PER_UNIT[axis.unit] / _SECONDS_PER_UNIT[unit]).apply(axis)
