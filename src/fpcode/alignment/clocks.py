"""Explicit clock calibration and time-unit conversion without fitting policy."""

from dataclasses import dataclass
import math

import numpy as np

from fpcode.contracts import TimeAxis

_SECONDS_PER_UNIT = {"s": 1.0, "ms": 1e-3, "ns": 1e-9}


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
        # Subtract integer origins before float conversion to preserve small
        # offsets in hardware timestamps around 10**18 nanoseconds.
        delta = axis.values - self.source_origin
        values = np.asarray(delta, dtype=float) * self.scale + self.target_origin
        return TimeAxis(values, self.target_unit, self.target_clock)


def convert_time_unit(axis: TimeAxis, unit: str, *, origin=0) -> TimeAxis:
    """Convert physical time units, optionally rebasing at an explicit origin.

    Frames require a calibrated ClockTransform; no FPS or clock is inferred.
    """
    if axis.unit not in _SECONDS_PER_UNIT or unit not in _SECONDS_PER_UNIT:
        raise ValueError("frame conversion requires an explicit clock calibration")
    return ClockTransform(axis.clock, axis.clock, axis.unit, unit, origin, 0,
                          _SECONDS_PER_UNIT[axis.unit] / _SECONDS_PER_UNIT[unit]).apply(axis)
