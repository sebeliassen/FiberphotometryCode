"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/audits/audit_20261007_independent_verification/audit_6_schedule_and_masking.py
==================================================
Relocated to: studies/behavior/multiview_experiments/04_reports_and_figures/audits/audit_20261007_independent_verification/audit_6_schedule_and_masking.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.04_reports_and_figures.audits.audit_20261007_independent_verification.audit_6_schedule_and_masking is deprecated; import from studies.behavior.multiview_experiments.04_reports_and_figures.audits.audit_20261007_independent_verification.audit_6_schedule_and_masking instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.04_reports_and_figures.audits.audit_20261007_independent_verification.audit_6_schedule_and_masking")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
