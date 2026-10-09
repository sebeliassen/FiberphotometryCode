"""
[DEPRECATED SHIM] scripts/launch_extended_when_limited_done.py
==================================================
Relocated to: ops/scripts/launch_extended_when_limited_done.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "scripts.launch_extended_when_limited_done is deprecated; import from ops.scripts.launch_extended_when_limited_done instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("ops.scripts.launch_extended_when_limited_done")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
