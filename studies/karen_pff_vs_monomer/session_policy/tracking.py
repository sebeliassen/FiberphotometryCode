"""
[DEPRECATED SHIM] studies/monoamine_interactions/projects/karen_project/tracking.py
==================================================
Relocated to: studies/behavior/karen_studies/tracking.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "studies.monoamine_interactions.projects.karen_project.tracking is deprecated; import from studies.behavior.karen_studies.tracking instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.tracking")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
