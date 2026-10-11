"""
[DEPRECATED SHIM] syllable_generation/analysis/syllable_chains/core.py
==================================================
Relocated to: studies/behavior/misc/syllable_generation/analysis/syllable_chains/core.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.syllable_chains.core is deprecated; import from studies.behavior.misc.syllable_generation.analysis.syllable_chains.core instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.misc.syllable_generation.analysis.syllable_chains.core")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
