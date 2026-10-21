"""Structural historical package retaining physical canonical leaf proxies."""

import importlib
import warnings

warnings.warn(
    "syllable_generation.loaders.jonathan.steps is deprecated; import from studies.session_building.jonathan.steps instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.session_building.jonathan.steps")


def __getattr__(name):
    return getattr(_real, name)


def __dir__():
    return dir(_real)
