"""
fpcode.common.paths
===================
Centralized Single Source of Truth for repository root and data path resolution.
Supports FP_DATA_ROOT environment variable with zero disk overhead fallback.
"""

import os
from pathlib import Path
from typing import Union

ENV_ROOT_VAR = "FIBERPHOTOMETRY_ROOT"
ENV_DATA_ROOT_VAR = "FP_DATA_ROOT"


def get_repo_root() -> Path:
    """
    Returns the absolute Path to the repository root.
    Checks FIBERPHOTOMETRY_ROOT environment variable first, then discovers
    via project anchor files (pyproject.toml / ARTIFACT_INVENTORY.tsv).
    """
    if ENV_ROOT_VAR in os.environ and os.environ[ENV_ROOT_VAR].strip():
        env_path = Path(os.environ[ENV_ROOT_VAR].strip()).resolve()
        if env_path.exists():
            return env_path

    # Discover upwards from current file
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / "ARTIFACT_INVENTORY.tsv").exists():
            return parent

    return Path("/home/fsp585/Desktop/FiberphotometryCode")


REPO_ROOT = get_repo_root()


def get_data_root() -> Path:
    """
    Returns the root path for data artifacts.
    Checks FP_DATA_ROOT environment variable, defaulting to repository root
    without requiring physical file relocations.
    """
    if ENV_DATA_ROOT_VAR in os.environ and os.environ[ENV_DATA_ROOT_VAR].strip():
        data_env = Path(os.environ[ENV_DATA_ROOT_VAR].strip()).resolve()
        if data_env.exists():
            return data_env
    return REPO_ROOT


DATA_ROOT = get_data_root()
CONFIGS_DIR = REPO_ROOT / "configs"
DOCS_DIR = REPO_ROOT / "docs"
TESTS_DIR = REPO_ROOT / "tests"
OUTPUTS_DIR = REPO_ROOT / "outputs"
ARCHIVE_DIR = REPO_ROOT / "archive"


def resolve_path(path_like: Union[str, Path]) -> Path:
    """Resolves a relative or absolute path against the project root."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def resolve_data_path(path_like: Union[str, Path]) -> Path:
    """Resolves data path against FP_DATA_ROOT."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (DATA_ROOT / p).resolve()


def to_relative_path(path_like: Union[str, Path]) -> str:
    """Converts any path within the repository into a clean workspace-relative string."""
    p = Path(path_like).resolve()
    try:
        rel = p.relative_to(REPO_ROOT.resolve())
        return str(rel)
    except ValueError:
        return str(path_like)


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensures a directory exists, creating parents if necessary."""
    p = resolve_path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p
