"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/01_data_and_fusion/test_4_golden_rules_fusion.py
==================================================
Relocated to: studies/behavior/multiview_experiments/01_data_and_fusion/test_4_golden_rules_fusion.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.01_data_and_fusion.test_4_golden_rules_fusion is deprecated; import from studies.behavior.multiview_experiments.01_data_and_fusion.test_4_golden_rules_fusion instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.01_data_and_fusion.test_4_golden_rules_fusion")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
