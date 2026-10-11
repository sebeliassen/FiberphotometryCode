"""
[DEPRECATED SHIM] syllable_generation/syllabizers/keypoint_moseq_syll/utils/compact_checkpoint.py
=================================================================================================
Relocated to: fpcode.kpms.checkpoints
"""

import importlib
import sys
import warnings

warnings.warn(
    "utils.compact_checkpoint is deprecated; import from fpcode.kpms.checkpoints instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("fpcode.kpms.checkpoints")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

compact_checkpoint = _real.compact_checkpoint
is_checkpoint_packed = _real.is_checkpoint_packed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compact KPMS checkpoint (shim)")
    parser.add_argument("source", help="Source checkpoint.h5")
    parser.add_argument("--dest", "-o", default=None, help="Destination checkpoint.h5")
    parser.add_argument("--mode", default="full", choices=["full", "slim"])
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    _real.compact_checkpoint(args.source, dest_path=args.dest, mode=args.mode, in_place=args.in_place)
