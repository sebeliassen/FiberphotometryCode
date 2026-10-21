"""Half-open event intervals and frame-indexed syllable bouts."""

import math
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class EventInterval:
    name: str
    start: float
    stop: float
    unit: str
    clock: str

    def __post_init__(self):
        if not math.isfinite(self.start) or not math.isfinite(self.stop) or self.stop < self.start:
            raise ValueError("event interval requires finite start <= stop")
        if self.unit not in {"s", "ms", "ns", "frame"} or not self.clock:
            raise ValueError("event interval requires an explicit time unit and clock")


@dataclass(frozen=True)
class Bout:
    syllable: int
    onset: int
    offset: int

    def __post_init__(self):
        if not all(isinstance(v, Integral) for v in (self.syllable, self.onset, self.offset)):
            raise ValueError("bout labels and frame bounds must be integers")
        if self.onset < 0 or self.offset <= self.onset:
            raise ValueError("bout requires 0 <= onset < offset")

    @property
    def duration(self):
        return self.offset - self.onset
