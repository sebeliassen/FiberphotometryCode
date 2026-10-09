"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/audits/audit_20261007_independent_verification/audit_5_heading_and_bouts_discrepancy.py
==================================================
Relocated to: studies/behavior/multiview_experiments/04_reports_and_figures/audits/audit_20261007_independent_verification/audit_5_heading_and_bouts_discrepancy.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.04_reports_and_figures.audits.audit_20261007_independent_verification.audit_5_heading_and_bouts_discrepancy is deprecated; import from studies.behavior.multiview_experiments.04_reports_and_figures.audits.audit_20261007_independent_verification.audit_5_heading_and_bouts_discrepancy instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.04_reports_and_figures.audits.audit_20261007_independent_verification.audit_5_heading_and_bouts_discrepancy")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
