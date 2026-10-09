"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/01_data_and_fusion/verify_fused_in_kpms_pipeline.py
==================================================
Relocated to: studies/behavior/multiview_experiments/01_data_and_fusion/verify_fused_in_kpms_pipeline.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.01_data_and_fusion.verify_fused_in_kpms_pipeline is deprecated; import from studies.behavior.multiview_experiments.01_data_and_fusion.verify_fused_in_kpms_pipeline instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.01_data_and_fusion.verify_fused_in_kpms_pipeline")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
