"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/generate_slide_friendly_figure1.py
==================================================
Relocated to: studies/behavior/sequence_viz/generate_slide_friendly_figure1.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.generate_slide_friendly_figure1 is deprecated; import from studies.behavior.sequence_viz.generate_slide_friendly_figure1 instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.generate_slide_friendly_figure1")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
