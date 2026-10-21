"""Stable recording identity without depending on historical Session classes."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SessionKey:
    recording: str
    subject: Optional[str] = None
    study: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.recording, str) or not self.recording.strip():
            raise ValueError("recording identity is required")
        for value in (self.subject, self.study):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("subject and study must be nonempty strings when provided")
