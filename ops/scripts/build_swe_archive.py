"""Compatibility alias for ops.repo.build_swe_archive."""
import importlib
import runpy
import sys
import warnings
from pathlib import Path

warnings.warn("ops.scripts.build_swe_archive is deprecated; use ops.repo.build_swe_archive",
              DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "repo/build_swe_archive.py"), run_name="__main__")
else:
    _real = importlib.import_module("ops.repo.build_swe_archive")
    sys.modules[__name__] = _real

    def __getattr__(name):
        return getattr(_real, name)

    def __dir__():
        return dir(_real)
