"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/render_1x3_final_exemplars.py
==================================================
Relocated to: studies/behavior/sequence_viz/render_1x3_final_exemplars.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.render_1x3_final_exemplars is deprecated; import from studies.behavior.sequence_viz.render_1x3_final_exemplars instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.render_1x3_final_exemplars")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
