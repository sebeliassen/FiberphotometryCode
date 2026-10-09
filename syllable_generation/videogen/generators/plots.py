"""
[DEPRECATED SHIM] syllable_generation/videogen/generators/plots.py
==================================================
Relocated to: studies/behavior/videogen/plots.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.videogen.generators.plots is deprecated; import from studies.behavior.videogen.plots instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.videogen.plots")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
