"""
[DEPRECATED SHIM] syllable_generation/analysis/karen_syllable_analysis/generate_top30_sig_dendrogram.py
==================================================
Relocated to: studies/behavior/karen_studies/generate_top30_sig_dendrogram.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.karen_syllable_analysis.generate_top30_sig_dendrogram is deprecated; import from studies.behavior.karen_studies.generate_top30_sig_dendrogram instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.generate_top30_sig_dendrogram")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
