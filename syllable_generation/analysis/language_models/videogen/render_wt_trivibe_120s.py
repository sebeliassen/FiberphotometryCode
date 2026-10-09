"""
[DEPRECATED SHIM] syllable_generation/analysis/language_models/videogen/render_wt_trivibe_120s.py
==================================================
Relocated to: studies/behavior/language_models/videogen/render_wt_trivibe_120s.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.language_models.videogen.render_wt_trivibe_120s is deprecated; import from studies.behavior.language_models.videogen.render_wt_trivibe_120s instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.language_models.videogen.render_wt_trivibe_120s")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
