"""Event-relative windows with explicit coordinates and validity masks."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from fpcode.alignment.resampling import align_signal
from fpcode.contracts import EventInterval, SignalMetadata, SignalSeries, TimeAxis
from fpcode.contracts.timelines import frozen_array, _shift_coordinates


@dataclass(frozen=True)
class EventWindows:
    events: Tuple[EventInterval, ...]
    offsets: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    signals: Tuple[SignalMetadata, ...]
    unit: str
    clock: str

    def __post_init__(self):
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "signals", tuple(self.signals))
        object.__setattr__(self, "offsets", frozen_array(self.offsets))
        object.__setattr__(self, "values", frozen_array(self.values))
        object.__setattr__(self, "valid", frozen_array(self.valid, dtype=bool))
        expected = (len(self.events), len(self.offsets), len(self.signals))
        if self.values.shape != expected or self.valid.shape != expected:
            raise ValueError("event windows require (events, offsets, channels) values and validity")
        if np.any(self.valid & ~np.isfinite(self.values)):
            raise ValueError("nonfinite event samples cannot be valid")


def extract_event_windows(signal, events, offsets, *, method="exact", tolerance=None, max_gap=None):
    """Align each event's start + offsets, preserving masks and source clock.

    Event bounds and offsets use the same unit as the signal. Window sampling
    is independent of event.stop; the caller chooses the requested offsets.
    """
    events = tuple(events)
    offsets = TimeAxis(offsets, signal.time.unit, "event_relative").values
    channels = len(signal.signals)
    values = np.empty((len(events), len(offsets), channels), dtype=float)
    validity = np.empty(values.shape, dtype=bool)
    for index, event in enumerate(events):
        if event.unit != signal.time.unit or event.clock != signal.time.clock:
            raise ValueError("event and signal time units and clock must match")
        axis = TimeAxis(_shift_coordinates(offsets, event.start), signal.time.unit, signal.time.clock)
        sampled = align_signal(signal, axis, method=method, tolerance=tolerance, max_gap=max_gap)
        values[index] = sampled.values[:, None] if sampled.values.ndim == 1 else sampled.values
        validity[index] = sampled.valid[:, None] if sampled.valid.ndim == 1 else sampled.valid
    return EventWindows(events, offsets, values, validity, signal.signals, signal.time.unit, signal.time.clock)


def legacy_nearest_window(timestamps, start, stop, *, expected_length=None):
    """Historical nearest-coordinate half-open slice with optional row length.

    This preserves argmin ties and clamped nearest endpoints. It performs no
    gap interpolation or epoch buffering; callers retain those study policies.
    """
    timestamps = np.asarray(timestamps)
    begin = int(np.abs(timestamps - start).argmin())
    end = int(np.abs(timestamps - stop).argmin())
    if expected_length is not None and end - begin != expected_length:
        end = begin + expected_length
    return begin, end


__all__ = ["EventWindows", "extract_event_windows", "legacy_nearest_window"]
