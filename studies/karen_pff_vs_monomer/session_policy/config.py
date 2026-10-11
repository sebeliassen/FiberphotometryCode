"""
[DEPRECATED SHIM] studies/monoamine_interactions/projects/karen_project/config.py
==================================================
Relocated to: src/fpcode/behavior/config.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "studies.monoamine_interactions.projects.karen_project.config is deprecated; import from fpcode.behavior.config instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.behavior.config")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
