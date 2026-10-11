"""Historical cohort export alias; canonical session composition lives in workflows."""

import importlib
import sys

_real = importlib.import_module("fpcode.compat.kpms_export")
sys.modules[__name__] = _real


def __getattr__(name):
    return getattr(_real, name)


def __dir__():
    return dir(_real)
