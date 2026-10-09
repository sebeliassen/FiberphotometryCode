"""
[DEPRECATED SHIM] syllable_generation/projects/jonathan_openfield_example/enrich.py
==================================================
Relocated to: studies/behavior/projects/enrich.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.projects.jonathan_openfield_example.enrich is deprecated; import from studies.behavior.projects.enrich instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.projects.enrich")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
