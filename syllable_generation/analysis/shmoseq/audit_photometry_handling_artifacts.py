"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/audit_photometry_handling_artifacts.py
==================================================
Relocated to: studies/behavior/shmoseq/audit_photometry_handling_artifacts.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.audit_photometry_handling_artifacts is deprecated; import from studies.behavior.shmoseq.audit_photometry_handling_artifacts instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.audit_photometry_handling_artifacts")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
