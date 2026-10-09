"""
[DEPRECATED SHIM] syllable_generation/projects/jonathan_openfield_amph/enrich.py
==================================================
Relocated to: studies/behavior/projects/enrich.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.projects.jonathan_openfield_amph.enrich is deprecated; import from studies.behavior.projects.enrich instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.projects.enrich")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
