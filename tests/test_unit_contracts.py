"""
tests/test_unit_contracts.py

Automated enforcement of docs/architecture/UNIT_CONTRACT.md:
1. Downward-only import direction: src/fpcode/ must NEVER import from studies, archive, or syllable_generation.
2. Core isolation: fpcode.common must not depend on domain logic (kpms, photometry, tracking, behavior).
3. Clean module importability: no top-level argparse.parse_args() or side-effects at import time.
"""

import ast
import importlib
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FPCODE_DIR = REPO_ROOT / "src" / "fpcode"

FORBIDDEN_UPWARD_IMPORTS = {
    "studies",
    "syllable_generation",
    "archive",
    "tests",
    "scratch",
}

KNOWN_STEP8_LEGACY_IMPORTS = set()

def get_all_fpcode_py_files():
    return sorted(list(FPCODE_DIR.rglob("*.py")))

def test_fpcode_downward_only_layering():
    """Verify that src/fpcode never imports upwards into studies, legacy, or archive."""
    violations = []

    for py_file in get_all_fpcode_py_files():
        try:
            tree = ast.parse(py_file.read_text(errors="ignore"), filename=str(py_file))
        except Exception:
            continue

        rel = str(py_file.relative_to(REPO_ROOT))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    if root_pkg in FORBIDDEN_UPWARD_IMPORTS:
                        if (rel, alias.name) not in KNOWN_STEP8_LEGACY_IMPORTS:
                            violations.append(f"{rel} imports forbidden upward package: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_pkg = node.module.split(".")[0]
                    if root_pkg in FORBIDDEN_UPWARD_IMPORTS:
                        if (rel, node.module) not in KNOWN_STEP8_LEGACY_IMPORTS:
                            violations.append(f"{rel} imports forbidden upward package: {node.module}")

    assert not violations, f"Downward import layering violated in src/fpcode:\n" + "\n".join(violations)

def test_fpcode_common_core_isolation():
    """Verify that fpcode.common never imports domain-specific logic."""
    common_dir = FPCODE_DIR / "common"
    domain_pkgs = {"fpcode.kpms", "fpcode.photometry", "fpcode.tracking", "fpcode.behavior"}
    violations = []

    for py_file in common_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(errors="ignore"), filename=str(py_file))
        except Exception:
            continue

        for node in ast.walk(tree):
            imported = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name.startswith(d) for d in domain_pkgs):
                        imported = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(node.module.startswith(d) for d in domain_pkgs):
                    imported = node.module

            if imported:
                rel = py_file.relative_to(REPO_ROOT)
                violations.append(f"{rel} imports domain package {imported}")

    assert not violations, f"fpcode.common isolation violated:\n" + "\n".join(violations)

def test_no_top_level_argparse_in_fpcode():
    """Verify that no module in src/fpcode runs parse_args() at top level (outside __main__)."""
    violations = []

    for py_file in get_all_fpcode_py_files():
        try:
            tree = ast.parse(py_file.read_text(errors="ignore"), filename=str(py_file))
        except Exception:
            continue

        for node in tree.body:
            # Check for top-level calls to parse_args()
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute) and call.func.attr == "parse_args":
                    rel = py_file.relative_to(REPO_ROOT)
                    violations.append(f"{rel} calls parse_args() at top level")

    assert not violations, f"Top-level parse_args() found:\n" + "\n".join(violations)

def test_fpcode_clean_importability():
    """Verify that all public submodules in fpcode can be imported without side-effects or errors."""
    submodules = [
        "fpcode.common.paths",
        "fpcode.common.schemas",
        "fpcode.common.h5_guard",
        "fpcode.common.logging",
        "fpcode.common.notify",
        "fpcode.common.run_record",
        "fpcode.common.cache_manifest",
        "fpcode.common.io",
        "fpcode.tracking.pair_matching",
        "fpcode.tracking.clock_sync",
        "fpcode.tracking.camera_correspondence",
        "fpcode.tracking.fusion",
        "fpcode.tracking.static_fusion",
        "fpcode.kpms.init_model",
        "fpcode.kpms.checkpoints",
        "fpcode.kpms.data",
        "fpcode.kpms.runtime",
        "fpcode.kpms.fit",
        "fpcode.kpms.export",
        "fpcode.kpms.metrics",
        "fpcode.kpms.evaluation",
        "fpcode.photometry",
        "fpcode.photometry.alignment",
    ]

    for mod_name in submodules:
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Failed to import {mod_name}"
