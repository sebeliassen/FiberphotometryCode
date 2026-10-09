"""Foundational records shared by acquisition adapters and domain algorithms."""

from .events import Bout, EventInterval
from .signals import SignalMetadata, SignalSeries
from .timelines import TimeAxis
from .pose import PoseSeries
from .sessions import SessionKey
from .artifacts import ArtifactReference, ArtifactSet

__all__ = ["TimeAxis", "SignalMetadata", "SignalSeries", "PoseSeries", "EventInterval", "Bout"]
