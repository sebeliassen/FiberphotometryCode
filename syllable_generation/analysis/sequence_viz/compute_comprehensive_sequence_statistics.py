"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/compute_comprehensive_sequence_statistics.py
==================================================
Relocated to: studies/behavior/sequence_viz/compute_comprehensive_sequence_statistics.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.compute_comprehensive_sequence_statistics is deprecated; import from studies.behavior.sequence_viz.compute_comprehensive_sequence_statistics instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.compute_comprehensive_sequence_statistics")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
