"""Transactional, reversible attribute patches for the optional KPMS backend."""

from contextlib import contextmanager


class PatchSet:
    """Capture each target's own prior value and restore it on failure or exit.

    Patches are process-global. Callers must serialize backend configuration;
    nested scopes within a single thread are supported.
    """

    def __init__(self, bindings):
        self.bindings = list(bindings)
        self._previous = []

    def apply(self):
        if self._previous:
            raise RuntimeError("This patch set is already installed")
        try:
            for target, name, replacement in self.bindings:
                previous = getattr(target, name)
                self._previous.append((target, name, previous))
                setattr(target, name, replacement)
        except BaseException:
            self.restore()
            raise
        return self

    def restore(self):
        previous, self._previous = self._previous, []
        error = None
        for target, name, value in reversed(previous):
            try:
                setattr(target, name, value)
            except BaseException as failure:
                error = error or failure
        if error is not None:
            raise error


@contextmanager
def temporary_patches(bindings):
    patches = PatchSet(bindings).apply()
    try:
        yield patches
    finally:
        patches.restore()
