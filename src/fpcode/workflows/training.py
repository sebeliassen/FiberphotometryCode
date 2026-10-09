"""Training orchestration with an explicit expensive backend boundary."""

from pathlib import Path
from typing import Callable, Mapping, Any

from fpcode.common.paths import resolve_path
from fpcode.kpms.config import resolve_training_config


def training_input_path(config: Mapping[str, Any], model_dir: Path) -> Path:
    return resolve_path(config["data_path"]) if config.get("data_path") else model_dir.parent / "preprocessed_data.pkl"


def validate_training_input(config: Mapping[str, Any], model_dir: Path) -> Path:
    path = training_input_path(config, model_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Training data not found at {path}. Prepare preprocessed_data.pkl first "
            "or supply data_path; data_dir alone is not a preprocessed training artifact."
        )
    return path


def fit_model(config: Mapping[str, Any], model_dir: Path, seed: int, device: str = "gpu", *, backend: Callable | None = None) -> Path:
    config = resolve_training_config(config)
    model_dir = Path(model_dir)
    config["data_path"] = str(validate_training_input(config, model_dir))
    if config.get("pca_path"):
        config["pca_path"] = str(resolve_path(config["pca_path"]))
    if backend is None:
        from fpcode.kpms.training import fit_model as backend
    checkpoint = Path(backend(config, model_dir, seed=seed, device=device))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training returned a missing checkpoint: {checkpoint}")
    return checkpoint
