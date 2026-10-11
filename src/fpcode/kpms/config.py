"""Validated training settings, independent of JAX and model fitting."""

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingSettings:
    num_ar_iters: int = 50
    num_slds_iters: int = 200
    save_every_n_iters: int = 25
    init_kappa: float | None = 1.0e6
    slds_kappa: float | None = 1.5e15
    use_memory_patch: bool = True
    use_batched_init: bool = True
    location_aware: bool = False

    @property
    def total_iters(self) -> int:
        return self.num_ar_iters + self.num_slds_iters


_PREPROCESSING_FIELDS = {
    "target_fps", "conf_threshold", "min_frames", "min_gap_frames",
    "min_segment_frames", "decrease_kappa_factor",
}
_FIELDS = set(TrainingSettings.__dataclass_fields__) | _PREPROCESSING_FIELDS | {"total_iters", "ar_kappa"}


def _integer(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def resolve_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize flat, ``hyperparameters`` and legacy ``training`` settings.

    Conflicting declarations fail rather than silently overriding one another.
    Preprocessing fields remain explicit metadata: this fitting entrypoint consumes
    already-preprocessed data and does not reinterpret its preparation policies.
    """
    if not isinstance(config, Mapping):
        raise ValueError("Training configuration must be a mapping")
    resolved = dict(config)
    declared = {key: config[key] for key in _FIELDS if key in config}
    for section_name in ("hyperparameters", "training"):
        section = config.get(section_name, {})
        if not isinstance(section, Mapping):
            raise ValueError(f"{section_name} must be a mapping")
        unknown = set(section) - _FIELDS
        if unknown:
            raise ValueError(f"Unknown {section_name} settings: {', '.join(sorted(unknown))}")
        for key, value in section.items():
            if key in declared and declared[key] != value:
                raise ValueError(f"Conflicting declarations of {key}")
            declared[key] = value

    values = asdict(TrainingSettings())
    values.update({key: value for key, value in declared.items() if key in values})
    if "init_kappa" not in declared and "ar_kappa" in declared:
        values["init_kappa"] = declared["ar_kappa"]
    for name in ("num_ar_iters", "num_slds_iters"):
        values[name] = _integer(name, values[name], 0)
    values["save_every_n_iters"] = _integer("save_every_n_iters", values["save_every_n_iters"], 1)
    for name in ("init_kappa", "slds_kappa"):
        value = values[name]
        if value is not None:
            if isinstance(value, bool):
                raise ValueError(f"{name} must be finite and nonnegative")
            value = float(value)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            values[name] = value
    for name in ("use_memory_patch", "use_batched_init", "location_aware"):
        if not isinstance(values[name], bool):
            raise ValueError(f"{name} must be a boolean")
    settings = TrainingSettings(**values)
    if settings.total_iters < 1:
        raise ValueError("Training must contain at least one iteration")
    if "total_iters" in declared and _integer("total_iters", declared["total_iters"], 1) != settings.total_iters:
        raise ValueError("total_iters must equal num_ar_iters + num_slds_iters")
    resolved.update(declared)
    resolved.update(asdict(settings))
    resolved["total_iters"] = settings.total_iters
    model = config.get("model", {})
    if isinstance(model, Mapping) and "model_name" not in resolved and "name" in model:
        resolved["model_name"] = model["name"]
    return resolved
