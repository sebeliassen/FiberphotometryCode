"""
[DEPRECATED SHIM] scripts/notify_kpms_iterations.py
==================================================
Relocated to: ops/scripts/notify_kpms_iterations.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "scripts.notify_kpms_iterations is deprecated; import from ops.scripts.notify_kpms_iterations instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("ops.scripts.notify_kpms_iterations")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
