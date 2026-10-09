"""
[DEPRECATED SHIM] scripts/relink_split_videos.py
==================================================
Relocated to: ops/scripts/relink_split_videos.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "scripts.relink_split_videos is deprecated; import from ops.scripts.relink_split_videos instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("ops.scripts.relink_split_videos")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
