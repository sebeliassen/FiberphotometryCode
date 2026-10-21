"""Foundational records shared by acquisition adapters and domain algorithms."""

from .artifacts import ArtifactReference, ArtifactSet
from .behavior_results import (
    BehaviorResult,
    CompletionEvidence,
    CompositionDerivation,
    FeatureAssociation,
    PhaseRecord,
    ScopedOccurrenceReference,
    StatePosteriorRecord,
)
from .behavior_units import FrameSpan, UnitOccurrence, UnitStream, UnitType
from .composition import (
    ChoiceRule,
    GrammarRule,
    GrammarSpecification,
    OptionalRule,
    ParseMatch,
    RepeatRule,
    RuleComponent,
    SequenceRule,
    TerminalMatch,
)
from .events import Bout, EventInterval
from .pose import PoseSeries
from .sessions import SessionKey
from .signals import SignalMetadata, SignalSeries
from .timelines import TimeAxis

__all__ = [
    "TimeAxis",
    "SignalMetadata",
    "SignalSeries",
    "PoseSeries",
    "EventInterval",
    "Bout",
    "SessionKey",
    "ArtifactReference",
    "ArtifactSet",
    "FrameSpan",
    "UnitType",
    "UnitOccurrence",
    "UnitStream",
    "ScopedOccurrenceReference",
    "FeatureAssociation",
    "PhaseRecord",
    "CompletionEvidence",
    "CompositionDerivation",
    "StatePosteriorRecord",
    "BehaviorResult",
    "TerminalMatch",
    "SequenceRule",
    "ChoiceRule",
    "OptionalRule",
    "RepeatRule",
    "RuleComponent",
    "GrammarRule",
    "GrammarSpecification",
    "ParseMatch",
]
