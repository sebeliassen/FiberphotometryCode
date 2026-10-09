"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/03_evaluation_and_eml/compare_k240k_vs_tandem_cohort.py
==================================================
Relocated to: studies/behavior/multiview_experiments/03_evaluation_and_eml/compare_k240k_vs_tandem_cohort.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.03_evaluation_and_eml.compare_k240k_vs_tandem_cohort is deprecated; import from studies.behavior.multiview_experiments.03_evaluation_and_eml.compare_k240k_vs_tandem_cohort instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.03_evaluation_and_eml.compare_k240k_vs_tandem_cohort")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
