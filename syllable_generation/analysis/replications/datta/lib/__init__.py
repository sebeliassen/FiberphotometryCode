"""
datta2 Library: Core logic for the Datta-style fiber photometry correlation analysis.
This library provides modularized functions for signal preprocessing, kinematics calculation, 
temporal binning, Pearson correlation analysis, and statistical shuffling.
"""
from importlib import import_module

__all__ = ["config", "features", "shuffling", "stats", "snippets"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = import_module(__name__ + "." + name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
