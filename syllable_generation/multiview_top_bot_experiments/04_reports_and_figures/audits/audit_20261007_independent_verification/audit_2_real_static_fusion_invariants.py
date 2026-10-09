"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/audits/audit_20261007_independent_verification/audit_2_real_static_fusion_invariants.py
==================================================
Relocated to: studies/behavior/multiview_experiments/04_reports_and_figures/audits/audit_20261007_independent_verification/audit_2_real_static_fusion_invariants.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.04_reports_and_figures.audits.audit_20261007_independent_verification.audit_2_real_static_fusion_invariants is deprecated; import from studies.behavior.multiview_experiments.04_reports_and_figures.audits.audit_20261007_independent_verification.audit_2_real_static_fusion_invariants instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.04_reports_and_figures.audits.audit_20261007_independent_verification.audit_2_real_static_fusion_invariants")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
