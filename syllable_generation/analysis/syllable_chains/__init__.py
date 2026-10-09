#!/usr/bin/env python3
"""
syllable_chains package
=============================================================================
Unified, cohort-agnostic framework for analyzing syllable chain dynamics:
Inter-chain organization (repeats, continuations, transitions, initiations)
and Intra-chain execution (consistency, completion, variations, pacing, context).
=============================================================================
"""

from .core import (
    ChainDefinition,
    ChainOccurrence,
    ChainBout,
    CohortDataAdapter,
    ChainFinder
)
from .inter_chain import InterChainAnalyzer
from .intra_chain import IntraChainAnalyzer
from .visualizer import ChainVisualizer

__all__ = [
    "ChainDefinition",
    "ChainOccurrence",
    "ChainBout",
    "CohortDataAdapter",
    "ChainFinder",
    "InterChainAnalyzer",
    "IntraChainAnalyzer",
    "ChainVisualizer"
]
