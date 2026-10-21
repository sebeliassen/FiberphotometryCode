"""Structural historical package retaining physical exporter utility proxies."""

import importlib
import warnings

warnings.warn(
    "syllable_generation.analysis.utils is deprecated; import from studies.session_model_exports.utils instead",
    DeprecationWarning,
    stacklevel=2,
)
_real = importlib.import_module("studies.session_model_exports.utils")


def __getattr__(name):
    return getattr(_real, name)


def __dir__():
    return dir(_real)
