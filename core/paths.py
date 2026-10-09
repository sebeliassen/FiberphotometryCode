"""
core.paths
==========
Centralized path resolution for FiberphotometryCode.
Eliminates hardcoded '/home/fsp585' strings by resolving relative to repo root
or reading the FIBERPHOTOMETRY_ROOT environment variable.
"""

import os
from pathlib import Path
from typing import Union

ENV_ROOT_VAR = "FIBERPHOTOMETRY_ROOT"


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

    # Walk upwards from current file to find anchor
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / "ARTIFACT_INVENTORY.tsv").exists():
            return parent

    # Fallback to current working directory or default
    return Path("/home/fsp585/Desktop/FiberphotometryCode")


REPO_ROOT = get_repo_root()
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
DOCS_DIR = REPO_ROOT / "docs"
TESTS_DIR = REPO_ROOT / "tests"
OUTPUTS_DIR = REPO_ROOT / "outputs"
MULTIVIEW_DIR = REPO_ROOT / "syllable_generation" / "multiview_top_bot_experiments"


def resolve_path(path_like: Union[str, Path]) -> Path:
    """Resolves a relative or absolute path against the project root."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def to_relative_path(path_like: Union[str, Path]) -> str:
    """
    Converts any path within the repository into a clean workspace-relative string.
    Ensures manifests and logs store portable relative paths.
    """
    p = Path(path_like).resolve()
    try:
        rel = p.relative_to(REPO_ROOT.resolve())
        return str(rel)
    except ValueError:
        # If outside repo root, return string as-is
        return str(path_like)


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensures a directory exists, creating parents if necessary."""
    p = resolve_path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p
