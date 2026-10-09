"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/generate_hidden_slide_mann_whitney_plot.py
==================================================
Relocated to: studies/behavior/sequence_viz/generate_hidden_slide_mann_whitney_plot.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.generate_hidden_slide_mann_whitney_plot is deprecated; import from studies.behavior.sequence_viz.generate_hidden_slide_mann_whitney_plot instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.generate_hidden_slide_mann_whitney_plot")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
