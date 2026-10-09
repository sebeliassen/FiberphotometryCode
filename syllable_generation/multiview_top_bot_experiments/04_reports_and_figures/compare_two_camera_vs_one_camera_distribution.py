"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/compare_two_camera_vs_one_camera_distribution.py
==================================================
Relocated to: studies/behavior/multiview_experiments/04_reports_and_figures/compare_two_camera_vs_one_camera_distribution.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.04_reports_and_figures.compare_two_camera_vs_one_camera_distribution is deprecated; import from studies.behavior.multiview_experiments.04_reports_and_figures.compare_two_camera_vs_one_camera_distribution instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.04_reports_and_figures.compare_two_camera_vs_one_camera_distribution")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
