"""Historical alias; numerical coordinate alignment lives in fpcode.alignment."""

import importlib
import sys

_real = importlib.import_module("fpcode.compat.photometry.frame_merge")
sys.modules[__name__] = _real


def __getattr__(name):
    return getattr(_real, name)


def __dir__():
    return dir(_real)
