"""
fpcode
======
Unified Fiberphotometry & Keypoint-MoSeq (KPMS) Research Software Suite.
The Single Source of Truth importable package.
"""

from . import behavior, common, kpms, photometry, tracking, viz

__version__ = "0.2.0"

__all__ = [
    "common",
    "tracking",
    "kpms",
    "photometry",
    "behavior",
    "viz",
    "__version__",
]
