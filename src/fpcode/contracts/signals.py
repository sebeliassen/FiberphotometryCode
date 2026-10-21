"""Signal coordinates, labels and sample validity without a session dependency."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .timelines import TimeAxis, frozen_array


@dataclass(frozen=True)
class SignalMetadata:
    name: str
    unit: str = "a.u."
    region: Optional[str] = None
    sensor: Optional[str] = None
    hemisphere: Optional[str] = None
    wavelength: Optional[str] = None


@dataclass(frozen=True)
class SignalSeries:
    time: TimeAxis
    values: np.ndarray
    signals: Tuple[SignalMetadata, ...]
    valid: Optional[np.ndarray] = None

    def __post_init__(self):
        values = frozen_array(self.values, dtype=float)
        if values.ndim not in (1, 2) or values.shape[0] != len(self.time):
            raise ValueError("signals must have shape (time,) or (time, channels)")
        signals = tuple(self.signals)
        channels = 1 if values.ndim == 1 else values.shape[1]
        if len(signals) != channels or len({s.name for s in signals}) != channels:
            raise ValueError("each channel requires a unique signal label")
        valid = frozen_array(np.isfinite(values) if self.valid is None else self.valid, dtype=bool)
        if valid.shape != values.shape or np.any(valid & ~np.isfinite(values)):
            raise ValueError("validity must match values and cannot mark nonfinite samples valid")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "valid", valid)
