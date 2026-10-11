"""
[DEPRECATED SHIM] syllable_generation/syllabizers/keypoint_moseq_syll/utils/patch_checkpoint.py
===============================================================================================
Relocated to: fpcode.kpms.checkpoints
"""

import importlib
import sys
import warnings

warnings.warn(
    "utils.patch_checkpoint is deprecated; import from fpcode.kpms.checkpoints instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.kpms.checkpoints")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

is_checkpoint_packed = _real.is_checkpoint_packed
patched_load_checkpoint = _real.patched_load_checkpoint
apply_kpms_checkpoint_patch = _real.apply_kpms_checkpoint_patch
remove_kpms_checkpoint_patch = _real.remove_kpms_checkpoint_patch
