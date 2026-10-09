"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/audits/audit_20261007_pipeline_repairs/repair_5_provenance_bound_cache_validator.py
==================================================
Relocated to: studies/behavior/multiview_experiments/04_reports_and_figures/audits/audit_20261007_pipeline_repairs/repair_5_provenance_bound_cache_validator.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.04_reports_and_figures.audits.audit_20261007_pipeline_repairs.repair_5_provenance_bound_cache_validator is deprecated; import from studies.behavior.multiview_experiments.04_reports_and_figures.audits.audit_20261007_pipeline_repairs.repair_5_provenance_bound_cache_validator instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.04_reports_and_figures.audits.audit_20261007_pipeline_repairs.repair_5_provenance_bound_cache_validator")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
