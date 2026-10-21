"""Compose explicit clock calibration, alignment and event-window extraction."""

from fpcode.alignment.resampling import align_signal
from fpcode.contracts import SignalSeries


def align_photometry(signal, target, *, clock_transform=None, method="exact", tolerance=None, max_gap=None):
    if clock_transform is not None:
        signal = SignalSeries(clock_transform.apply(signal.time), signal.values, signal.signals, signal.valid)
    return align_signal(signal, target, method=method, tolerance=tolerance, max_gap=max_gap)


def event_photometry(signal, events, offsets, *, clock_transform=None,
                     method="exact", tolerance=None, max_gap=None):
    from fpcode.photometry.event_windows import extract_event_windows

    if clock_transform is not None:
        signal = SignalSeries(clock_transform.apply(signal.time), signal.values, signal.signals, signal.valid)
    return extract_event_windows(signal, events, offsets, method=method, tolerance=tolerance, max_gap=max_gap)
