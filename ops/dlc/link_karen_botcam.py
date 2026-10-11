"""
[DEPRECATED SHIM] scripts/link_karen_botcam.py
==================================================
Relocated to: studies/karen_pff_vs_monomer/analysis/link_karen_botcam.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "scripts.link_karen_botcam is deprecated; import from studies.karen_pff_vs_monomer.analysis.link_karen_botcam instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.karen_pff_vs_monomer.analysis.link_karen_botcam")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
