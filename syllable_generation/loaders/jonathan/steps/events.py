"""
[DEPRECATED SHIM] syllable_generation/loaders/jonathan/steps/events.py
==================================================
Relocated to: studies/behavior/loaders/events.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.loaders.jonathan.steps.events is deprecated; import from studies.behavior.loaders.events instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.loaders.events")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
