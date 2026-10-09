"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/capacity/run_detailed_audit_and_timelines.py
==================================================
Relocated to: studies/behavior/shmoseq/capacity/run_detailed_audit_and_timelines.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.capacity.run_detailed_audit_and_timelines is deprecated; import from studies.behavior.shmoseq.capacity.run_detailed_audit_and_timelines instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.capacity.run_detailed_audit_and_timelines")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
