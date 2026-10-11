"""Explicit artifact references; construction never reads or writes a file."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class ArtifactReference:
    name: str
    path: Path
    kind: str
    sha256: Optional[str] = None

    def __post_init__(self):
        if not self.name or not self.kind:
            raise ValueError("artifact name and kind are required")
        object.__setattr__(self, "path", Path(self.path))
        if self.sha256 is not None and (
            len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256)
        ):
            raise ValueError("sha256 must be a lowercase 64-character digest")


@dataclass(frozen=True)
class ArtifactSet:
    set_id: str
    artifacts: Tuple[ArtifactReference, ...]

    def __post_init__(self):
        artifacts = tuple(self.artifacts)
        if not self.set_id or not artifacts or len({a.name for a in artifacts}) != len(artifacts):
            raise ValueError("artifact sets require an identity and uniquely named members")
        object.__setattr__(self, "artifacts", artifacts)
