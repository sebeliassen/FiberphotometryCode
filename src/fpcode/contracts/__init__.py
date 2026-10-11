"""Foundational records shared by acquisition adapters and domain algorithms."""

from .artifacts import ArtifactReference, ArtifactSet
from .events import Bout, EventInterval
from .pose import PoseSeries
from .sessions import SessionKey
from .signals import SignalMetadata, SignalSeries
from .timelines import TimeAxis

__all__ = ["TimeAxis", "SignalMetadata", "SignalSeries", "PoseSeries", "EventInterval", "Bout",
           "SessionKey", "ArtifactReference", "ArtifactSet"]
