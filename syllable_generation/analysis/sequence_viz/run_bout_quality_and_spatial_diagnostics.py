"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/run_bout_quality_and_spatial_diagnostics.py
==================================================
Relocated to: studies/behavior/sequence_viz/run_bout_quality_and_spatial_diagnostics.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.run_bout_quality_and_spatial_diagnostics is deprecated; import from studies.behavior.sequence_viz.run_bout_quality_and_spatial_diagnostics instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.run_bout_quality_and_spatial_diagnostics")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
