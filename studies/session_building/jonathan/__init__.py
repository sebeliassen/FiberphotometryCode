"""Jonathan session contracts and an explicitly requested build entry point."""

from .context import BuildContext

__all__ = ["BuildContext", "build", "build_sessions"]


def __dir__():
    return sorted(set(globals()) | set(__all__))


def __getattr__(name):
    # Configuration, parsers and serialized classes must be usable independently
    # of the builder's project enrichment and optional workflow dependencies.
    # The build submodule can bind the package's historical `build` attribute.
    # `build_sessions` has no colliding child module, so it remains the same
    # callable regardless of package/leaf import order. Preserve the old export
    # for external notebooks while giving new callers a stable lazy entry point.
    if name in ("build", "build_sessions"):
        from .build import build

        globals()[name] = build
        return build
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
