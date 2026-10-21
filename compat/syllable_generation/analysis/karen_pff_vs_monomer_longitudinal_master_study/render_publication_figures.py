"""
[DEPRECATED SHIM] syllable_generation/analysis/karen_pff_vs_monomer_longitudinal_master_study/render_publication_figures.py
==================================================
Relocated to: studies/behavior/karen_studies/render_publication_figures.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.karen_pff_vs_monomer_longitudinal_master_study.render_publication_figures is deprecated; import from studies.behavior.karen_studies.render_publication_figures instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.render_publication_figures")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
