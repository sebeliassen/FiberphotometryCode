"""Explicit compatibility defaults for the historical multiview cohort.

The JSON files remain packaged beside the historical tracking entry points.
Selecting them here does not read data, discover sessions, or write a cache.
New callers should pass their own resources to the tracking functions.
"""

import json
from collections.abc import MutableMapping
from pathlib import Path

from fpcode.common.paths import get_repo_root, resolve_data_path

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "tracking"
TTL_CACHE_PATH = RESOURCE_DIR / "ttl_cohort_offsets.json"
STATIC_TRANSFORMS_PATH = RESOURCE_DIR / "static_arena_transforms.json"
EPOCH_TRANSFORMS_PATH = RESOURCE_DIR / "epoch_arena_transforms.json"

ARENA_TO_BOTCAM = {f"Arena{i}": f"BotCam{i}" for i in range(1, 5)}
ARENA_TO_CHAMBER = {"Arena1": "top_left", "Arena2": "bottom_left", "Arena3": "bottom_right"}


def legacy_timestamp_root():
    """Historical source selection, used only by explicitly requested discovery."""
    mounted = Path("/mnt/hdd/karen_project")
    return mounted if mounted.exists() else resolve_data_path("karen_project")


def legacy_generated_ttl_path():
    """Generated caches belong in caller output, never the installed package."""
    return get_repo_root() / "outputs" / "multiview_compatibility" / "ttl_cohort_offsets.json"


def legacy_tracking_paths():
    """Keep the old checkout path policy behind deliberate legacy calls."""
    from fpcode.common.schemas import PathConfig
    return PathConfig()


def legacy_session_epoch(timestamp):
    if timestamp < "2026-04-01":
        return "epoch1_winter_spring_2026"
    return "epoch2_spring_post_shift_2026"


class LazyJsonMapping(MutableMapping):
    """Mutable historical mapping without opening calibration JSON on import."""

    def __init__(self, path, *, missing_ok=False):
        self.path, self.missing_ok, self._data = Path(path), missing_ok, None

    def _load(self):
        if self._data is None:
            if self.missing_ok and not self.path.exists():
                self._data = {}
            else:
                with self.path.open() as handle:
                    self._data = json.load(handle)
        return self._data

    def __getitem__(self, key):
        return self._load()[key]

    def __setitem__(self, key, value):
        self._load()[key] = value

    def __delitem__(self, key):
        del self._load()[key]

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())


_STATIC_TRANSFORMS = LazyJsonMapping(STATIC_TRANSFORMS_PATH)
_EPOCH_TRANSFORMS = LazyJsonMapping(EPOCH_TRANSFORMS_PATH, missing_ok=True)


def legacy_static_calibration(arena, timestamp, *, use_epochs=False):
    """Resolve the old epoch/global/Arena1 fallback policy in study context."""
    if use_epochs and _EPOCH_TRANSFORMS:
        epoch = legacy_session_epoch(timestamp)
        transform = _EPOCH_TRANSFORMS.get(epoch, {}).get(arena)
        if transform is not None:
            return transform, EPOCH_TRANSFORMS_PATH, epoch
        resolved_epoch = "fallback_static"
    else:
        resolved_epoch = "static_global"
    transform = _STATIC_TRANSFORMS.get(arena, _STATIC_TRANSFORMS.get("Arena1"))
    return transform, STATIC_TRANSFORMS_PATH, resolved_epoch
