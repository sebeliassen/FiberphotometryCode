"""
datta2 Library: Core logic for the Datta-style fiber photometry correlation analysis.
This library provides modularized functions for signal preprocessing, kinematics calculation, 
temporal binning, Pearson correlation analysis, and statistical shuffling.
"""
from importlib import import_module

# Configuration and statistical helpers retain their historical locations and
# data/output paths. Resolve them explicitly rather than importing a bare lib.
_MODULES = {
    "config": "syllable_generation.analysis.replications.datta.lib.config",
    "stats": "syllable_generation.analysis.replications.datta.lib.stats",
    "features": __name__ + ".features",
    "shuffling": __name__ + ".shuffling",
    "snippets": __name__ + ".snippets",
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = import_module(_MODULES[name])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
