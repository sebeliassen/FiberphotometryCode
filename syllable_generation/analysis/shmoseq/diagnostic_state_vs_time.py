"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/diagnostic_state_vs_time.py
==================================================
Relocated to: studies/behavior/shmoseq/diagnostic_state_vs_time.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.diagnostic_state_vs_time is deprecated; import from studies.behavior.shmoseq.diagnostic_state_vs_time instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.diagnostic_state_vs_time")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
