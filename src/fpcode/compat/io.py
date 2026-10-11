"""Historical I/O namespace; checkpoint operations are owned by KPMS."""

from fpcode.common import file_io as _generic
from fpcode.kpms.checkpoint_io import compact_kpms_checkpoint, inspect_checkpoint_iterations

__all__ = ["compact_kpms_checkpoint", "inspect_checkpoint_iterations"]

# Preserve the historical utility namespace, including explicit private imports.
for _name, _value in vars(_generic).items():
    if not _name.startswith("__"):
        globals().setdefault(_name, _value)


def __dir__():
    return sorted(set(globals()) | set(dir(_generic)))
