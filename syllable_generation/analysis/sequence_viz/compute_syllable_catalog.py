"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/compute_syllable_catalog.py
==================================================
Relocated to: studies/behavior/sequence_viz/compute_syllable_catalog.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.compute_syllable_catalog is deprecated; import from studies.behavior.sequence_viz.compute_syllable_catalog instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.compute_syllable_catalog")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
