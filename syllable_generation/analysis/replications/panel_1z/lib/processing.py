"""
[DEPRECATED SHIM] syllable_generation/analysis/replications/panel_1z/lib/processing.py
==================================================
Relocated to: studies/behavior/replications_datta/panel_1z/lib/processing.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.replications.panel_1z.lib.processing is deprecated; import from studies.behavior.replications_datta.panel_1z.lib.processing instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.replications_datta.panel_1z.lib.processing")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
