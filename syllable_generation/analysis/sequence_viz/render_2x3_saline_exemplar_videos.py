"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/render_2x3_saline_exemplar_videos.py
==================================================
Relocated to: studies/behavior/sequence_viz/render_2x3_saline_exemplar_videos.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.render_2x3_saline_exemplar_videos is deprecated; import from studies.behavior.sequence_viz.render_2x3_saline_exemplar_videos instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.render_2x3_saline_exemplar_videos")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
