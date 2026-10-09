"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/render_1x3_theme_B_right_turn_options.py
==================================================
Relocated to: studies/behavior/sequence_viz/render_1x3_theme_B_right_turn_options.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.render_1x3_theme_B_right_turn_options is deprecated; import from studies.behavior.sequence_viz.render_1x3_theme_B_right_turn_options instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.render_1x3_theme_B_right_turn_options")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
