"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/render_syllable_12_random_movie.py
==================================================
Relocated to: studies/behavior/multiview_experiments/04_reports_and_figures/render_syllable_12_random_movie.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.04_reports_and_figures.render_syllable_12_random_movie is deprecated; import from studies.behavior.multiview_experiments.04_reports_and_figures.render_syllable_12_random_movie instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.04_reports_and_figures.render_syllable_12_random_movie")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
