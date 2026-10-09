"""
fpcode.common.config
====================
Centralized YAML configuration parser and hasher.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .paths import resolve_path


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads and parses a YAML configuration file."""
    p = resolve_path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {p}")
    return data


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 hash over sorted JSON serialization of config."""
    serialized = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
