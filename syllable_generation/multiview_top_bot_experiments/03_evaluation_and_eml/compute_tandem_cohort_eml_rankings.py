"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/03_evaluation_and_eml/compute_tandem_cohort_eml_rankings.py
==================================================
Relocated to: studies/behavior/multiview_experiments/03_evaluation_and_eml/compute_tandem_cohort_eml_rankings.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.03_evaluation_and_eml.compute_tandem_cohort_eml_rankings is deprecated; import from studies.behavior.multiview_experiments.03_evaluation_and_eml.compute_tandem_cohort_eml_rankings instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.03_evaluation_and_eml.compute_tandem_cohort_eml_rankings")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
