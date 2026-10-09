"""
[DEPRECATED SHIM] scripts/build_swe_archive.py
==================================================
Relocated to: ops/scripts/build_swe_archive.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "scripts.build_swe_archive is deprecated; import from ops.scripts.build_swe_archive instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("ops.scripts.build_swe_archive")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
