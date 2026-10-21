"""Explicit landmark and file inputs shared by visualization helpers."""

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class LandmarkGroups:
    """Column positions and labels for a pose representation's QC groups."""

    dorsal: tuple[int, ...]
    paws: tuple[int, ...]
    paw_names: tuple[str, ...]
    front_paws: tuple[int, ...] = ()
    hind_paws: tuple[int, ...] = ()

    def __post_init__(self):
        for name in ("dorsal", "paws", "front_paws", "hind_paws", "paw_names"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        indexes = self.dorsal + self.paws + self.front_paws + self.hind_paws
        if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indexes):
            raise ValueError("Landmark indexes must be nonnegative integers")
        if not self.dorsal or not self.paws or len(self.paws) != len(self.paw_names):
            raise ValueError("QC requires dorsal points and named paw points")
        if len(set(self.paws)) != len(self.paws) or len(set(self.paw_names)) != len(self.paw_names):
            raise ValueError("Paw indexes and names must be unique")
        if set(self.dorsal) & set(self.paws):
            raise ValueError("Dorsal and paw QC groups must be distinct")
        if not set(self.front_paws + self.hind_paws) <= set(self.paws):
            raise ValueError("Front and hind paw groups must use named paw points")


@dataclass(frozen=True)
class BottomPoseLandmarks:
    """Landmark names and fallback location for the bottom-view pose cascade."""

    nose: str
    tail: str
    front_paws: tuple[str, ...]
    hind_paws: tuple[str, ...]
    fallback_centroid: tuple[float, float]

    def __post_init__(self):
        for name in ("front_paws", "hind_paws", "fallback_centroid"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.fallback_centroid) != 2:
            raise ValueError("The fallback centroid must contain two coordinates")


@dataclass(frozen=True)
class SelectionInputs:
    """Provenance inputs chosen by a study or workflow, without discovery."""

    results: Path
    config: Path
    video_frame_indexes: Path
    preprocessed: Path
    calibration_files: tuple[Path, ...] = ()
    fused_artifacts: Mapping[str, Path] = field(default_factory=dict)
    schema_version: int = 4

    def __post_init__(self):
        for name in ("results", "config", "video_frame_indexes", "preprocessed"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        object.__setattr__(self, "calibration_files", tuple(Path(path) for path in self.calibration_files))
        object.__setattr__(self, "fused_artifacts", MappingProxyType({
            str(key): Path(path) for key, path in self.fused_artifacts.items()
        }))
