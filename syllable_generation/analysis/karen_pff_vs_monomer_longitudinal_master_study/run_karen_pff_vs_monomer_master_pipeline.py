"""
[DEPRECATED SHIM] syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/run_karen_pff_vs_monomer_master_pipeline.py
==================================================
Relocated to: studies/behavior/karen_studies/run_karen_pff_vs_monomer_master_pipeline.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.karen_pff_vs_monomer_longitudinal_master_study.run_karen_pff_vs_monomer_master_pipeline is deprecated; import from studies.behavior.karen_studies.run_karen_pff_vs_monomer_master_pipeline instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.run_karen_pff_vs_monomer_master_pipeline")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
