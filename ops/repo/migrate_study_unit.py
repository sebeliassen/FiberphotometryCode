#!/usr/bin/env python3
"""
ops/repo/migrate_study_unit.py

Automates migrating a study unit adhering to docs/architecture/UNIT_CONTRACT.md:
- Relocates files from source path to target studies/<project>/
- Leaves transparent backward-compatibility shims at old paths emitting DeprecationWarning
- Generates standard README.md and __init__.py
- Appends to registry/SHIMS.tsv and registry/MOVES.tsv
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

SHIM_TEMPLATE = '''"""
[DEPRECATED SHIM] {src_rel}
==================================================
Relocated to: {dest_rel}
"""

import importlib
import sys
import warnings

warnings.warn(
    "{src_mod} is deprecated; import from {dest_mod} instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("{dest_mod}")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
'''

def path_to_module(rel_path: str) -> str:
    p = Path(rel_path)
    parts = list(p.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)

def migrate_unit(
    unit_name: str,
    dest_study_dir: str,
    file_mappings: List[Tuple[str, str]], # (src_rel, dest_rel)
    title: str,
    purpose: str,
    status: str = "ACTIVE",
    inputs: str = "Preprocessed cohort recordings or h5 pose tracks",
    outputs: str = "Figures and summary metrics under outputs/ or $FP_DATA_ROOT",
    how_to_run: str = "python run.py --config config.yaml",
):
    os.chdir(REPO_ROOT)
    dest_dir = REPO_ROOT / dest_study_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    lines_moved = 0
    shims_added = 0
    moves_records = []
    shims_records = []

    for src_rel, dest_rel in file_mappings:
        src_path = REPO_ROOT / src_rel
        dest_path = REPO_ROOT / dest_rel

        if not src_path.exists():
            continue

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        content = src_path.read_text(errors="ignore")
        num_lines = len(content.splitlines())
        lines_moved += num_lines

        # 1. Copy content to destination
        dest_path.write_text(content)
        moved_count += 1

        # 2. Replace source with deprecation shim if .py file
        if src_rel.endswith(".py") and not src_rel.endswith("__init__.py"):
            src_mod = path_to_module(src_rel)
            dest_mod = path_to_module(dest_rel)
            shim_content = SHIM_TEMPLATE.format(
                src_rel=src_rel,
                dest_rel=dest_rel,
                src_mod=src_mod,
                dest_mod=dest_mod,
            )
            src_path.write_text(shim_content)
            shims_added += 1
            shims_records.append((src_rel, dest_mod, "DeprecationWarning", "ACTIVE"))

        moves_records.append((src_rel, dest_rel, "STUDY_MIGRATION", f"Migrated {unit_name} study file", "SHIM"))

    # 3. Create or update README.md in destination study
    readme_path = dest_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(f"""# {title}

**Location**: `{dest_study_dir}`  
**Status**: `{status}`  
**Standard**: Conforms to [`docs/architecture/UNIT_CONTRACT.md`](../../docs/architecture/UNIT_CONTRACT.md)  

## 1. Purpose
{purpose}

## 2. Inputs
{inputs}

## 3. Outputs
{outputs}

## 4. How to Run
```bash
{how_to_run}
```
""")

    # 4. Create or update __init__.py in destination study
    init_path = dest_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text(f'"""{title} package."""\n')

    # 5. Append records
    with open(REPO_ROOT / "registry" / "MOVES.tsv", "a") as f:
        for r in moves_records:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\n")

    with open(REPO_ROOT / "registry" / "SHIMS.tsv", "a") as f:
        for r in shims_records:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\n")

    print(f"Migrated unit '{unit_name}': {moved_count} files ({lines_moved} lines), {shims_added} shims installed.")
    return moved_count, lines_moved, shims_added
