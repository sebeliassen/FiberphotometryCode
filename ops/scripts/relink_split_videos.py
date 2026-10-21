"""Compatibility alias for ops.dlc.relink_split_videos."""
import importlib
import runpy
import sys
import warnings
from pathlib import Path

warnings.warn("ops.scripts.relink_split_videos is deprecated; use ops.dlc.relink_split_videos",
              DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "dlc/relink_split_videos.py"), run_name="__main__")
else:
    _real = importlib.import_module("ops.dlc.relink_split_videos")
    sys.modules[__name__] = _real

    def __getattr__(name):
        return getattr(_real, name)

    def __dir__():
        return dir(_real)
