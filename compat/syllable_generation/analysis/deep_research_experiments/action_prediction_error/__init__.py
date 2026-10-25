"""Structural compatibility package: retain physical leaf proxies and identity."""

import importlib
import warnings

warnings.warn(
    "syllable_generation.analysis.deep_research_experiments.action_prediction_error is deprecated; "
    "import from studies.deep_research_experiments.action_prediction_error instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.deep_research_experiments.action_prediction_error")
# Keep this package's physical search path: aliasing the parent to the namespace
# would bypass leaf proxies and execute their implementations under old names.


def __getattr__(name):
    return getattr(_real, name)


def __dir__():
    return dir(_real)
