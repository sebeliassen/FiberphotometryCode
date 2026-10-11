"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/audit_dlc_injection_timing.py
==================================================
Relocated to: studies/behavior/shmoseq/audit_dlc_injection_timing.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.audit_dlc_injection_timing is deprecated; import from studies.behavior.shmoseq.audit_dlc_injection_timing instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.audit_dlc_injection_timing")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
