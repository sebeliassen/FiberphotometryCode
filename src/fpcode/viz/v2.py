"""Historical V2 facade; defaults and wrappers live in fpcode.compat.viz."""

import fpcode.compat.viz.v2 as _facade

__all__ = _facade.__all__


def __getattr__(name):
    return getattr(_facade, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_facade)))
