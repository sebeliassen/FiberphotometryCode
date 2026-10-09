"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/compare_shmoseq_stickiness_8kp_vs_17kp.py
==================================================
Relocated to: studies/behavior/shmoseq/compare_shmoseq_stickiness_8kp_vs_17kp.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.compare_shmoseq_stickiness_8kp_vs_17kp is deprecated; import from studies.behavior.shmoseq.compare_shmoseq_stickiness_8kp_vs_17kp instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.compare_shmoseq_stickiness_8kp_vs_17kp")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
