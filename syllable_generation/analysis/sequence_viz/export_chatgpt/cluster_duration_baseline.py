"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/export_chatgpt/cluster_duration_baseline.py
==================================================
Relocated to: studies/behavior/sequence_viz/export_chatgpt/cluster_duration_baseline.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.export_chatgpt.cluster_duration_baseline is deprecated; import from studies.behavior.sequence_viz.export_chatgpt.cluster_duration_baseline instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.export_chatgpt.cluster_duration_baseline")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
