"""
[DEPRECATED SHIM] scripts/ntfy_monitor.py
==================================================
Relocated to: ops/scripts/ntfy_monitor.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "scripts.ntfy_monitor is deprecated; import from ops.scripts.ntfy_monitor instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("ops.scripts.ntfy_monitor")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
