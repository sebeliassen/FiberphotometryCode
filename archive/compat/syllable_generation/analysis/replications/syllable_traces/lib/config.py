"""
[DEPRECATED SHIM] syllable_generation/analysis/replications/syllable_traces/lib/config.py
==================================================
Relocated to: studies/behavior/replications_datta/syllable_traces/lib/config.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.replications.syllable_traces.lib.config is deprecated; import from studies.behavior.replications_datta.syllable_traces.lib.config instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.replications_datta.syllable_traces.lib.config")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
