"""
[DEPRECATED SHIM] syllable_generation/analysis/render_all_run12_visualizations.py
==================================================
Relocated to: studies/behavior/misc/syllable_generation/analysis/render_all_run12_visualizations.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.render_all_run12_visualizations is deprecated; import from studies.behavior.misc.syllable_generation.analysis.render_all_run12_visualizations instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.misc.syllable_generation.analysis.render_all_run12_visualizations")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
