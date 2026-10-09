"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/capacity/audit_shmoseq_evidence.py
==================================================
Relocated to: studies/behavior/shmoseq/capacity/audit_shmoseq_evidence.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.capacity.audit_shmoseq_evidence is deprecated; import from studies.behavior.shmoseq.capacity.audit_shmoseq_evidence instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.capacity.audit_shmoseq_evidence")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
