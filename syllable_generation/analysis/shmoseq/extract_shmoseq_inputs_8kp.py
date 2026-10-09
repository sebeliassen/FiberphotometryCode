"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/extract_shmoseq_inputs_8kp.py
==================================================
Relocated to: studies/behavior/shmoseq/extract_shmoseq_inputs_8kp.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.extract_shmoseq_inputs_8kp is deprecated; import from studies.behavior.shmoseq.extract_shmoseq_inputs_8kp instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.extract_shmoseq_inputs_8kp")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
