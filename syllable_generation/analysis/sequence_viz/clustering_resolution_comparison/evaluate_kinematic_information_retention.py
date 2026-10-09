"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/clustering_resolution_comparison/evaluate_kinematic_information_retention.py
==================================================
Relocated to: studies/behavior/sequence_viz/clustering_resolution_comparison/evaluate_kinematic_information_retention.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.clustering_resolution_comparison.evaluate_kinematic_information_retention is deprecated; import from studies.behavior.sequence_viz.clustering_resolution_comparison.evaluate_kinematic_information_retention instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.clustering_resolution_comparison.evaluate_kinematic_information_retention")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
