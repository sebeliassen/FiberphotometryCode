"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/generate_reality_check_slide_figures.py
==================================================
Relocated to: studies/behavior/shmoseq/generate_reality_check_slide_figures.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.generate_reality_check_slide_figures is deprecated; import from studies.behavior.shmoseq.generate_reality_check_slide_figures instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.generate_reality_check_slide_figures")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
