"""
fpcode
======
Unified Fiberphotometry & Keypoint-MoSeq (KPMS) Research Software Suite.
The Single Source of Truth importable package.
"""

from importlib import import_module

__version__ = "0.2.0"

__all__ = [
    "common",
    "tracking",
    "kpms",
    "photometry",
    "behavior",
    "viz",
    "contracts",
    "adapters",
    "alignment",
    "statistics",
    "workflows",
    "cli",
    "__version__",
]


def __getattr__(name):
    """Load capabilities only when requested; base imports initialize no backends."""
    if name in __all__ and name != "__version__":
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
