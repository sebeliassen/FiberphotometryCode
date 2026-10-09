"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/02_training_runners/run_tandem_cohort_production_50ar_200slds.py
==================================================
Relocated to: studies/behavior/multiview_experiments/02_training_runners/run_tandem_cohort_production_50ar_200slds.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.02_training_runners.run_tandem_cohort_production_50ar_200slds is deprecated; import from studies.behavior.multiview_experiments.02_training_runners.run_tandem_cohort_production_50ar_200slds instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.02_training_runners.run_tandem_cohort_production_50ar_200slds")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
