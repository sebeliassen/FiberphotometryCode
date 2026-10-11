"""
[DEPRECATED SHIM] studies/monoamine_interactions/projects/karen_project/enrich.py
==================================================
Relocated to: studies/behavior/karen_studies/enrich.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "studies.monoamine_interactions.projects.karen_project.enrich is deprecated; import from studies.behavior.karen_studies.enrich instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.enrich")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
