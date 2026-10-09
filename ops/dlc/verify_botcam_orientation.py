"""
[DEPRECATED SHIM] scripts/verify_botcam_orientation.py
==================================================
Relocated to: ops/scripts/verify_botcam_orientation.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "scripts.verify_botcam_orientation is deprecated; import from ops.scripts.verify_botcam_orientation instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("ops.scripts.verify_botcam_orientation")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
