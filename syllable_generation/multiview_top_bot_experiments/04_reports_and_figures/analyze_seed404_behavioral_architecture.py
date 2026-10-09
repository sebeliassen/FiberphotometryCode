"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/analyze_seed404_behavioral_architecture.py
==================================================
Relocated to: studies/behavior/multiview_experiments/04_reports_and_figures/analyze_seed404_behavioral_architecture.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.04_reports_and_figures.analyze_seed404_behavioral_architecture is deprecated; import from studies.behavior.multiview_experiments.04_reports_and_figures.analyze_seed404_behavioral_architecture instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.04_reports_and_figures.analyze_seed404_behavioral_architecture")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
