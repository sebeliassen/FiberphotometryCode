"""
[DEPRECATED SHIM] syllable_generation/analysis/replications/datta/lib/stats.py
==================================================
Relocated to: studies/behavior/replications_datta/datta/lib/stats.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.replications.datta.lib.stats is deprecated; import from studies.replications_datta.datta.lib.stats instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.replications_datta.datta.lib.stats")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
