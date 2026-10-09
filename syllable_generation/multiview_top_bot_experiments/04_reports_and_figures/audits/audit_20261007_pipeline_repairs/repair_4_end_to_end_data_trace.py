"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/audits/audit_20261007_pipeline_repairs/repair_4_end_to_end_data_trace.py
==================================================
Relocated to: studies/behavior/multiview_experiments/04_reports_and_figures/audits/audit_20261007_pipeline_repairs/repair_4_end_to_end_data_trace.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.04_reports_and_figures.audits.audit_20261007_pipeline_repairs.repair_4_end_to_end_data_trace is deprecated; import from studies.behavior.multiview_experiments.04_reports_and_figures.audits.audit_20261007_pipeline_repairs.repair_4_end_to_end_data_trace instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.04_reports_and_figures.audits.audit_20261007_pipeline_repairs.repair_4_end_to_end_data_trace")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
