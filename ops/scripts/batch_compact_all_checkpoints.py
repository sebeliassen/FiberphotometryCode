"""Compatibility alias for ops.compaction.batch_compact_all_checkpoints."""
import importlib
import runpy
import sys
import warnings
from pathlib import Path

warnings.warn("ops.scripts.batch_compact_all_checkpoints is deprecated; use ops.compaction.batch_compact_all_checkpoints",
              DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "compaction/batch_compact_all_checkpoints.py"), run_name="__main__")
else:
    _real = importlib.import_module("ops.compaction.batch_compact_all_checkpoints")
    sys.modules[__name__] = _real

    def __getattr__(name):
        return getattr(_real, name)

    def __dir__():
        return dir(_real)
