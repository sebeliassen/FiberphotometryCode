"""Jonathan session contracts and an explicitly requested build entry point."""

from .context import BuildContext

__all__ = ["BuildContext", "build"]


def __dir__():
    return sorted(set(globals()) | set(__all__))


def __getattr__(name):
    # Configuration, parsers and serialized classes must be usable independently
    # of the builder's project enrichment and optional workflow dependencies.
    if name == "build":
        from .build import build

        globals()[name] = build
        return build
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
