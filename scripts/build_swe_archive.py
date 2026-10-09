#!/usr/bin/env python3
"""
Build lightweight code-review zip archive for FiberphotometryCode repository.
Replaces large non-selfwritten files (data, binaries, models, videos, logs, etc.)
with informative .txt placeholders, strips Jupyter notebook outputs,
and includes an executive SWE review prompt and archive README.
"""

import json
import mimetypes
import os
import time
import zipfile
from datetime import datetime, timezone

WORKSPACE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
OUTPUT_ZIP_PATH = os.path.join(WORKSPACE_DIR, "fiberphotometry_code_review.zip")

EXCLUDE_DIR_NAMES = {".git", "__pycache__", ".pytest_cache"}

ALWAYS_CODE_EXTS = {
    ".py", ".md", ".rst", ".sh", ".bash", ".zsh",
    ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".dockerignore", "dockerfile"
}

ALWAYS_PLACEHOLDER_EXTS = {
    ".csv", ".tsv", ".parquet", ".feather", ".h5", ".hdf5",
    ".pkl", ".pickle", ".p", ".joblib", ".pt", ".pth", ".onnx",
    ".npy", ".npz",
    ".avi", ".mp4", ".mov", ".mkv",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg", ".pdf",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z",
    ".log", ".out", ".pid", ".wandb", ".db", ".sqlite", ".sqlite3",
    ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll"
}

MAX_TEXT_SIZE_BYTES = 500 * 1024       # 500 KB limit for .txt, .json
MAX_HTML_SIZE_BYTES = 200 * 1024       # 200 KB limit for .html


def format_bytes(num_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def get_file_type_desc(fname: str, ext: str) -> str:
    ext_lower = ext.lower()
    descriptions = {
        ".csv": "Tabular Data / Comma-Separated Values (CSV)",
        ".tsv": "Tabular Data / Tab-Separated Values (TSV)",
        ".parquet": "Apache Parquet Columnar Storage",
        ".feather": "Feather Binary Arrow Storage",
        ".h5": "HDF5 Hierarchical Data Container (DeepLabCut / Keypoint-MoSeq)",
        ".hdf5": "HDF5 Hierarchical Data Container",
        ".pkl": "Python Pickle Serialized Object / Model Checkpoint",
        ".pickle": "Python Pickle Serialized Object",
        ".p": "Python Pickle Serialized Object",
        ".joblib": "Joblib Serialized Model / Array",
        ".pt": "PyTorch Tensor / Model Weights",
        ".pth": "PyTorch Model Checkpoint",
        ".onnx": "ONNX Model Binary",
        ".npy": "NumPy Array Binary Dump",
        ".npz": "NumPy Zipped Multi-Array Archive",
        ".avi": "Video Recording (AVI)",
        ".mp4": "Video Recording (MP4)",
        ".mov": "Video Recording (QuickTime)",
        ".png": "Raster Image (PNG / Figure / Diagnostic)",
        ".jpg": "Raster Image (JPEG)",
        ".jpeg": "Raster Image (JPEG)",
        ".gif": "Animated GIF / Diagnostic Render",
        ".pdf": "PDF Document / Figure Export",
        ".zip": "ZIP Compressed Archive / Backup",
        ".tar": "TAR Archive",
        ".gz": "Gzip Compressed Archive",
        ".log": "Execution Log / Terminal Output",
        ".out": "Standard Output Dump (nohup / cluster)",
        ".wandb": "Weights & Biases Run Artifact / Cache",
        ".db": "Database File",
        ".sqlite": "SQLite Database",
    }
    if ext_lower in descriptions:
        return descriptions[ext_lower]

    mime_type, _ = mimetypes.guess_type(fname)
    if mime_type:
        return f"{mime_type} ({ext_lower})"
    return f"Binary / Data File ({ext_lower if ext_lower else 'no extension'})"


def strip_notebook_outputs(nb_content_str: str) -> str:
    try:
        nb_json = json.loads(nb_content_str)
        cells = nb_json.get("cells", [])
        for cell in cells:
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        return json.dumps(nb_json, indent=1)
    except Exception:
        # If parsing fails, return as-is
        return nb_content_str


def make_placeholder_content(rel_path: str, fname: str, ext: str, size: int, mtime: float, is_symlink: bool, symlink_target: str, is_broken: bool) -> str:
    try:
        dt_local = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        dt_utc = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception:
        dt_local = "Unknown"
        dt_utc = "Unknown"

    type_desc = get_file_type_desc(fname, ext)
    human_size = format_bytes(size)

    content = [
        "================================================================================",
        "[PLACEHOLDER FILE - ORIGINAL REPLACED TO OPTIMIZE ARCHIVE SIZE]",
        "================================================================================",
        f"Original Filename      : {fname}",
        f"Original Relative Path : {rel_path}",
        f"Detected File Type     : {type_desc}",
        f"File Extension         : {ext if ext else '(none)'}",
        f"Original Size (Bytes)  : {size:,} bytes",
        f"Original Size (Human)  : {human_size}",
        f"Last Modified (Local)  : {dt_local}",
        f"Last Modified (UTC)    : {dt_utc}",
        f"Is Symlink             : {is_symlink}",
    ]
    if is_symlink:
        content.append(f"Symlink Target         : {symlink_target}")
        content.append(f"Symlink Status         : {'BROKEN' if is_broken else 'VALID'}")

    content.append("================================================================================")
    content.append("Note for AI / SWE Reviewers:")
    content.append("This file represents a real dataset, model checkpoint, video recording, or output")
    content.append("artifact in the repository. Its binary content has been replaced with this metadata")
    content.append("placeholder so you have full visibility into the directory structure, data formats,")
    content.append("file sizes, and timestamps without exceeding context or upload limits.")
    content.append("================================================================================")
    content.append("")
    return "\n".join(content)


def generate_swe_bot_prompt(stats: dict) -> str:
    return f"""# Software Engineering & Architecture Review Prompt

You are acting as a Principal Research Software Engineer and Python Architect specializing in scientific computing, machine learning pipelines, and computational neuroscience.

You are provided with a complete code review archive (`fiberphotometry_code_review.zip`) of the `FiberphotometryCode` repository.

---

## 1. Repository Background & Scientific Domain
This codebase supports advanced behavioral neuroscience and fiber photometry analysis:
- **Keypoint-MoSeq (KPMS)**: Unsupervised behavioral segmentation of animal postures into discrete syllables using Autoregressive Hidden Markov Models (ARHMM) and SLDS.
- **Multi-Camera & Tandem View**: Synchronized top and bottom camera tracking (DeepLabCut 21/23 keypoints), affine camera registration, pseudo-3D posture modeling, and behavioral syllable classification.
- **Fiber Photometry Analysis**: Synchronized calcium / monoamine sensor signal processing (e.g. DA, 5-HT, NE), baseline correction, peri-event time histograms (PETHs), and behavioral correlation.
- **Sequence Analysis**: Ethological sequence analysis (SeqNMF, syllable chains, transition matrices).

---

## 2. Archive Construction & Placeholder Convention
To allow full repository review without gigabytes of raw video, DLC tracking data, and model checkpoints:
- **Source Code**: Python files (`.py`), Markdown (`.md`), Shell scripts (`.sh`), and configuration files (`.yaml`, `.json`, etc.) are included **verbatim** in full.
- **Jupyter Notebooks**: All `.ipynb` notebooks are included with execution cell outputs stripped (retaining all markdown and Python code cells).
- **Data & Binaries as Placeholders**: All large non-selfwritten files (CSVs, HDF5 `.h5`, Pickles `.pkl`, video recordings `.avi`/`.mp4`, plots `.png`, and zip backups) are replaced with **`<original_filename>.txt`** placeholder files. Each placeholder records:
  - Original filename and relative path
  - Detected file type / format
  - Exact byte size and human-readable size
  - Last modified timestamp
  - Symlink details (if applicable)

### Summary Statistics of this Archive:
- **Total Files Represented**: {stats['total_files']:,}
- **Verbatim Code & Config Files**: {stats['code_files']:,}
- **Stripped Jupyter Notebooks**: {stats['notebook_files']:,}
- **Lightweight Placeholders**: {stats['placeholder_files']:,} (representing {stats['placeholder_orig_gb']:.2f} GB of raw data/binaries)

---

## 3. Your Review Objectives & Expected Deliverables

Please conduct a thorough, rigorous software engineering audit of this repository and provide an actionable, prioritized roadmap across the following 6 core pillars:

### Pillar 1: Repository Structure & Directory Hierarchy
- Identify organizational anti-patterns (e.g., redundant backup folders like `syllable_generation_backup_*`, `multiview_top_bot_experiments_code_backup_*`, root-level clutter, orphaned scratch folders).
- Propose a clean, standardized monorepo or multi-package layout following modern Python standards (e.g., `src/`, `configs/`, `pipelines/`, `scripts/`, `docs/`, `tests/`).
- Recommend proper segregation between **raw input data**, **cached intermediate features**, **model checkpoints**, and **final presentation figures/deliverables**.

### Pillar 2: Architectural Modularity & Separation of Concerns
- Analyze script vs library boundaries: many scripts mix CLI arguments, data loading, heavy mathematical fitting, visualization generation, and file I/O in single monolithic files.
- Design reusable module boundaries (e.g., `io`, `calibration`, `modeling`, `metrics`, `visualization`).
- Address hardcoded assumptions (e.g., absolute filesystem paths `/home/fsp585/...` vs configuration management using Hydra, Pydantic, or YAML).

### Pillar 3: Code Duplication & Dead Code Pruning
- Identify cloned or slightly diverged duplicate scripts across different experiment directories.
- Identify legacy prototypes that should be archived or merged into canonical runners.
- Highlight common utility functions that should be consolidated into a shared core package.

### Pillar 4: Reproducibility, Configuration & Environment Management
- Audit dependencies, requirements, and environment management (e.g., `pyproject.toml`, conda `environment.yml`).
- Recommend best practices for hardware-accelerated scientific computing (GPU/CUDA memory scaling, page cache compaction, PyTorch/JAX memory management).
- Propose data artifact tracking (e.g., DVC or lightweight manifest hashing) so experiments can reference inputs/checkpoints cleanly.

### Pillar 5: Testing, Validation & Continuous Integration
- Review the current `tests/` directory (noting generated test run outputs vs actual unit/integration tests).
- Design a test strategy including:
  - Fast unit tests for geometry/affine math and photometry signal processing.
  - Mocked integration tests for KPMS model fitting and pipeline runners.
  - Regression tests for syllable classification and trajectory generation.
- Suggest a lightweight GitHub Actions / CI pipeline.

### Pillar 6: Prioritized Step-by-Step Refactoring Roadmap
Provide a phased implementation plan:
- **Phase 1: Immediate Hygiene & Quick Wins** (safe file moves, `.gitignore` hardening, root cleanup, removing dead backups).
- **Phase 2: Core Package Consolidation** (establishing canonical packages, shared utilities, and config schema).
- **Phase 3: Pipeline & CLI Standardization** (refactoring runners to use unified CLI/config interfaces).
- **Phase 4: CI/CD & Testing Infrastructure**.

---
Please begin your audit by analyzing the top-level repository structure and identifying the highest-impact architectural improvements.
"""


def generate_archive_readme(stats: dict) -> str:
    return f"""# FiberphotometryCode Repository Code Review Archive

This archive was created for automated software engineering analysis, architectural review, and codebase reorganization.

## Archive Metadata
- **Creation Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
- **Source Workspace**: `FiberphotometryCode`
- **Total Files in Repository**: {stats['total_files']:,}
- **Source Code & Config Files (Full)**: {stats['code_files']:,}
- **Jupyter Notebooks (Outputs Stripped)**: {stats['notebook_files']:,}
- **Data & Binary Placeholders (.txt)**: {stats['placeholder_files']:,}
- **Original Data & Binary Footprint**: {stats['placeholder_orig_gb']:.2f} GB

## Placeholder Convention
Files that are large datasets, model weights, videos, plots, or archives are stored as:
`[original_filename].txt`

Each placeholder contains complete metadata about the file's original type, size in bytes, human-readable size, and modification timestamp.

## Review Instructions for AI / SWE Bot
See `REPOSITORY_STRUCTURE_AND_PROMPT.md` at the root of this archive for detailed background and review objectives.
"""


def main():
    print("=" * 80)
    print("STARTING REPOSITORY ARCHIVE CREATION")
    print(f"Source Workspace: {WORKSPACE_DIR}")
    print(f"Destination Zip : {OUTPUT_ZIP_PATH}")
    print("=" * 80)

    start_time = time.time()

    # Pre-scan files
    stats = {
        'total_files': 0,
        'code_files': 0,
        'notebook_files': 0,
        'placeholder_files': 0,
        'placeholder_orig_gb': 0.0,
    }

    files_to_process = []

    for root, dirs, files in os.walk(WORKSPACE_DIR):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES]
        rel_root = os.path.relpath(root, WORKSPACE_DIR)
        if any(part in EXCLUDE_DIR_NAMES for part in rel_root.split(os.sep)):
            continue

        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, WORKSPACE_DIR)

            # Skip target zip itself or temporary files
            if os.path.abspath(full_path) == os.path.abspath(OUTPUT_ZIP_PATH):
                continue
            if f.endswith(".zip.tmp"):
                continue

            files_to_process.append((full_path, rel_path, f))

    print(f"Total candidate files to package: {len(files_to_process):,}")

    temp_zip_path = OUTPUT_ZIP_PATH + ".tmp"
    if os.path.exists(temp_zip_path):
        os.remove(temp_zip_path)

    with zipfile.ZipFile(temp_zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for full_path, rel_path, f in files_to_process:
            stats['total_files'] += 1
            ext = os.path.splitext(f)[1].lower()
            fname_lower = f.lower()

            # Inspect file stat safely without breaking on dead symlinks
            is_symlink = os.path.islink(full_path)
            symlink_target = ""
            is_broken = False
            mtime = 0.0
            size = 0

            if is_symlink:
                try:
                    symlink_target = os.readlink(full_path)
                    is_broken = not os.path.exists(full_path)
                    st = os.lstat(full_path)
                    mtime = st.st_mtime
                    size = st.st_size
                    # If target exists and not broken, get target size
                    if not is_broken:
                        size = os.path.getsize(full_path)
                        mtime = os.path.getmtime(full_path)
                except Exception:
                    is_broken = True
            else:
                try:
                    st = os.stat(full_path)
                    mtime = st.st_mtime
                    size = st.st_size
                except Exception:
                    size = 0
                    mtime = 0.0

            # Determine whether this is code or placeholder
            is_code = False
            is_notebook = False

            if ext == ".ipynb":
                is_notebook = True
            elif ext in ALWAYS_CODE_EXTS or fname_lower in {'license', '.gitignore', 'dockerfile'}:
                is_code = True
            elif ext in {'.json', '.txt'}:
                is_code = (size < MAX_TEXT_SIZE_BYTES)
            elif ext == '.html':
                is_code = (size < MAX_HTML_SIZE_BYTES)
            elif ext in ALWAYS_PLACEHOLDER_EXTS:
                is_code = False
            else:
                # Fallback for unknown extension: small plain text check
                if size < 50 * 1024 and not is_symlink:
                    try:
                        with open(full_path, 'rb') as cf:
                            chunk = cf.read(1024)
                            is_code = (b'\0' not in chunk)
                    except Exception:
                        is_code = False
                else:
                    is_code = False

            # Write to zip
            if is_notebook:
                stats['notebook_files'] += 1
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='replace') as nbf:
                        content = nbf.read()
                    stripped_content = strip_notebook_outputs(content)
                    zf.writestr(rel_path, stripped_content)
                except Exception:
                    # Fallback to placeholder if unreadable
                    stats['notebook_files'] -= 1
                    stats['placeholder_files'] += 1
                    stats['placeholder_orig_gb'] += size / (1024 ** 3)
                    placeholder_name = rel_path + ".txt"
                    content = make_placeholder_content(rel_path, f, ext, size, mtime, is_symlink, symlink_target, is_broken)
                    zf.writestr(placeholder_name, content)

            elif is_code and not is_symlink:
                stats['code_files'] += 1
                try:
                    zf.write(full_path, arcname=rel_path)
                except Exception:
                    # In case of read error, write placeholder
                    stats['code_files'] -= 1
                    stats['placeholder_files'] += 1
                    stats['placeholder_orig_gb'] += size / (1024 ** 3)
                    placeholder_name = rel_path + ".txt"
                    content = make_placeholder_content(rel_path, f, ext, size, mtime, is_symlink, symlink_target, is_broken)
                    zf.writestr(placeholder_name, content)

            else:
                stats['placeholder_files'] += 1
                stats['placeholder_orig_gb'] += size / (1024 ** 3)
                placeholder_name = rel_path + ".txt"
                content = make_placeholder_content(rel_path, f, ext, size, mtime, is_symlink, symlink_target, is_broken)
                zf.writestr(placeholder_name, content)

            if stats['total_files'] % 2500 == 0:
                print(f"Processed {stats['total_files']:,} files...")

        # Add README and Prompt to the root of the archive
        readme_text = generate_archive_readme(stats)
        prompt_text = generate_swe_bot_prompt(stats)
        zf.writestr("README_CODE_REVIEW_ARCHIVE.md", readme_text)
        zf.writestr("REPOSITORY_STRUCTURE_AND_PROMPT.md", prompt_text)

    # Atomically rename
    if os.path.exists(OUTPUT_ZIP_PATH):
        os.remove(OUTPUT_ZIP_PATH)
    os.rename(temp_zip_path, OUTPUT_ZIP_PATH)

    elapsed = time.time() - start_time
    zip_size_mb = os.path.getsize(OUTPUT_ZIP_PATH) / (1024 * 1024)

    print("=" * 80)
    print("ARCHIVE CREATION COMPLETE!")
    print(f"Output File          : {OUTPUT_ZIP_PATH}")
    print(f"Final Zip Size       : {zip_size_mb:.2f} MB")
    print(f"Total Files Handled  : {stats['total_files']:,}")
    print(f"Code Files (Verbatim): {stats['code_files']:,}")
    print(f"Notebooks (Stripped) : {stats['notebook_files']:,}")
    print(f"Placeholders Created : {stats['placeholder_files']:,}")
    print(f"Original Data Size   : {stats['placeholder_orig_gb']:.2f} GB represented as metadata")
    print(f"Elapsed Time         : {elapsed:.2f} seconds")
    print("=" * 80)

    # Also save prompt to workspace for direct user access
    prompt_file_path = os.path.join(WORKSPACE_DIR, "SWE_ORGANIZATION_BOT_PROMPT.md")
    with open(prompt_file_path, "w", encoding="utf-8") as pf:
        pf.write(prompt_text)
    print(f"Saved Prompt File    : {prompt_file_path}")


if __name__ == "__main__":
    main()
