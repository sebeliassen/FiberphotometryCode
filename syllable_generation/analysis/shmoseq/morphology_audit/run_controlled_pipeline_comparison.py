"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/morphology_audit/run_controlled_pipeline_comparison.py
==================================================
Relocated to: studies/behavior/shmoseq/morphology_audit/run_controlled_pipeline_comparison.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.morphology_audit.run_controlled_pipeline_comparison is deprecated; import from studies.behavior.shmoseq.morphology_audit.run_controlled_pipeline_comparison instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.morphology_audit.run_controlled_pipeline_comparison")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
