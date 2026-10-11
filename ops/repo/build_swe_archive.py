#!/usr/bin/env python3
"""
Build lightweight code-review zip archive for FiberphotometryCode repository.
Replaces large non-selfwritten files (data, binaries, models, videos, logs, etc.)
with informative .txt placeholders, strips Jupyter notebook outputs,
and includes the comprehensive 3-day Refactor Report and an executive SWE review prompt.
"""

import json
import mimetypes
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE_DIR = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[2]))
OUTPUT_ZIP_PATH = os.path.join(WORKSPACE_DIR, "archive", "bundles", "fiberphotometry_code_review.zip")

EXCLUDE_DIR_NAMES = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}

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
    return f"""# Scientific Python repository review

Review the provided source snapshot of FiberphotometryCode. Read
`docs/README.md`, `docs/architecture/UNIT_CONTRACT.md`, and
`docs/reports/refactor_report.md` for current guidance and scoped historical
observations. `REFACTOR_REPORT.md` is an intentional root export of that report,
not a second maintained repository document.

Current ownership: shared capabilities in `src/fpcode`; project methods and
scientific policy in `studies/<project>`; operational implementations in
purpose-specific `ops` directories; live historical imports in `compat`; frozen
material in `archive`; empirical inputs/models in `data`; outputs in `outputs`.
Study organization is by project, without separate behavior/photometry roots.

Assess actual evidence rather than treating stated invariance requirements as
completed empirical proof. Functional fixture tests and the bounded two-session
GPU diagnostic establish specific checked behaviors. They do not establish
universal pickle compatibility, scientific convergence, full-cohort memory
behavior or universal numerical equivalence. Backup coverage applies only to the
included snapshot inventory; read exclusions and symlink semantics in recovery.md.
Historical reports preserve earlier observations and may use earlier paths.

The archive contains {stats['code_files']:,} verbatim source/config files,
{stats['notebook_files']:,} notebooks with outputs stripped, and
{stats['placeholder_files']:,} metadata placeholders. Data/binaries and large
text files are represented by `<filename>.txt`. File and directory symlinks are
represented by metadata placeholders and their targets are not traversed.
Small registry TSVs are included verbatim for migration review.

Prioritize findings about broken imports/literal filesystem references, duplicate
mechanisms, dependency ownership, explicit scientific policy, import side effects,
meaningful independent tests and packaging. Separate verified failures, evidence
limitations and optional design suggestions. Static source alone cannot establish
that every historical script runs on empirical data.
"""


def generate_archive_readme(stats: dict) -> str:
    return f"""# FiberphotometryCode Repository Code Review Archive

This archive was created for automated software engineering analysis, architectural review, and codebase reorganization.

## Archive Metadata
- **Creation Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
- **Source Workspace**: `FiberphotometryCode`
- **Working Branch**: `codex/modular-refactor` (PR #1)
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
See `REFACTOR_REPORT.md` at the root of this archive for the 3-day refactor report.
"""


def main():
    print("=" * 80)
    print("STARTING REPOSITORY ARCHIVE CREATION")
    print(f"Source Workspace: {WORKSPACE_DIR}")
    print(f"Destination Zip : {OUTPUT_ZIP_PATH}")
    print("=" * 80)

    Path(OUTPUT_ZIP_PATH).parent.mkdir(parents=True, exist_ok=True)
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
        directory_links = [d for d in dirs if os.path.islink(os.path.join(root, d))]
        for directory_link in directory_links:
            full_path = os.path.join(root, directory_link)
            files_to_process.append((full_path, os.path.relpath(full_path, WORKSPACE_DIR), directory_link))
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES and d not in directory_links]
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
            elif ext == '.tsv' and rel_path.split(os.sep)[0] == 'registry':
                is_code = (size < MAX_TEXT_SIZE_BYTES)
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

        # Include the newly created Refactor Report right at the root of the zip archive!
        refactor_report_path = os.path.join(WORKSPACE_DIR, "docs", "reports", "refactor_report.md")
        if os.path.exists(refactor_report_path):
            with open(refactor_report_path, "r", encoding="utf-8") as rrf:
                zf.writestr("REFACTOR_REPORT.md", rrf.read())

    # Atomically rename
    if os.path.exists(OUTPUT_ZIP_PATH):
        os.remove(OUTPUT_ZIP_PATH)
    os.rename(temp_zip_path, OUTPUT_ZIP_PATH)

    elapsed = time.time() - start_time
    zip_size_mb = os.path.getsize(OUTPUT_ZIP_PATH) / (1024 * 1024)

    print("=" * 80)
    print("ARCHIVE CREATION COMPLETE!")
    print(f"Output Archive      : {OUTPUT_ZIP_PATH}")
    print(f"Final Zip Size       : {zip_size_mb:.2f} MB")
    print(f"Total Files Handled  : {stats['total_files']:,}")
    print(f"Code Files (Verbatim): {stats['code_files']:,}")
    print(f"Notebooks (Stripped) : {stats['notebook_files']:,}")
    print(f"Placeholders Created : {stats['placeholder_files']:,}")
    print(f"Original Data Size   : {stats['placeholder_orig_gb']:.2f} GB represented as metadata")
    print(f"Elapsed Time         : {elapsed:.2f} seconds")
    print("=" * 80)

    # Also save prompt to workspace for direct user access
    prompt_file_path = os.path.join(WORKSPACE_DIR, "archive", "bundles", "fiberphotometry_code_review_prompt.md")
    with open(prompt_file_path, "w", encoding="utf-8") as pf:
        pf.write(prompt_text)
    print(f"Saved Prompt File    : {prompt_file_path}")


if __name__ == "__main__":
    main()
