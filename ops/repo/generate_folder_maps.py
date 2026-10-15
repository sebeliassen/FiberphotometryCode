"""Hierarchical Bottom-Up Summarization & Architecture Map Generator.

Deterministically extracts module docstrings, symbols, dependencies, and tags
using Python's native AST parser (zero hallucination, zero extra dependencies).
Generates .folder-map.md files level-by-level from leaf packages up to the root,
and produces ARCHITECTURE_INDEX.md at the repository root.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]

# Target directory roots to include in hierarchical indexing
DEFAULT_TARGETS = (
    "src/fpcode",
    "pipelines",
    "ops",
    "configs",
    "tests",
    "studies",
)

# Directories that must NEVER be traversed or indexed
EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "cache",
    "caches",
    "session_caches",
    "raw_data",
    "tracking_data",
    "h5_files",
    "videos",
    "avis",
    "archive",
    "outputs",
    "output",
    "data",
    "vendor",
    "registry",
    ".vscode",
    ".idea",
    "wandb",
    ".wandb",
    "run_records",
    "scratch",
}

# Curated tags for canonical modules in src/fpcode
CANONICAL_TAGS: dict[str, list[str]] = {
    "fpcode": ["computational-neuroscience", "fiberphotometry", "keypoint-moseq", "tracking", "pipelines"],
    "fpcode.contracts": ["data-models", "protocols", "timelines", "pose", "session-records"],
    "fpcode.common": ["io", "logging", "caching", "schemas", "paths"],
    "fpcode.adapters": ["data-adapters", "legacy-sessions", "multiview-profile", "kpms-results"],
    "fpcode.alignment": ["temporal-alignment", "resampling", "frame-mapping", "interpolation"],
    "fpcode.statistics": ["summaries", "bootstrap", "smoothing", "signal-processing"],
    "fpcode.behavior": ["kinematics", "bouts", "heading", "segmentation", "ethology"],
    "fpcode.behavior.sequences": ["markov-chains", "transition-matrices", "sequence-mining"],
    "fpcode.tracking": ["3d-tracking", "multiview-fusion", "camera-calibration", "registration"],
    "fpcode.photometry": ["fiberphotometry", "dff-signals", "event-windows", "response-metrics"],
    "fpcode.kpms": ["keypoint-moseq", "slds", "ar-hmm", "checkpoint-io", "training", "jax"],
    "fpcode.kpms.backend": ["moseq-backend", "batching", "state-management", "gpu-acceleration"],
    "fpcode.kpms.backend.kernels": ["autoregression-kernels", "discrete-kernels", "continuous-kernels", "cuda"],
    "fpcode.viz": ["grid-movies", "trajectories", "pose-overlays", "figures", "matplotlib"],
    "fpcode.viz.selection": ["exemplar-selection", "bout-filtering", "density-sampling"],
    "fpcode.workflows": ["orchestration", "batch-pipeline", "session-export", "multimodal-analysis"],
    "fpcode.cli": ["cli-tools", "entrypoints", "argparse", "subcommands"],
    "fpcode.compat": ["backward-compatibility", "historical-shims", "legacy-unpickling"],
    "fpcode.compat.multiview": ["multiview-legacy-profile", "camera-sync-shims"],
    "fpcode.compat.photometry": ["acquisition-alignment-shims", "frame-merge-facade"],
    "fpcode.compat.viz": ["v2-viz-compatibility", "legacy-plotting-defaults"],
}

# Curated purposes for canonical packages when __init__.py docstrings are minimal
CANONICAL_PURPOSES: dict[str, str] = {
    "fpcode": "Unified Fiberphotometry & Keypoint-MoSeq (KPMS) research software suite; single source of truth library.",
    "fpcode.contracts": "Foundational dataclass records, timeline coordinates, pose series, and serialization protocols.",
    "fpcode.common": "Low-level utilities for structured logging, filesystem I/O, cache manifests, and configuration validation.",
    "fpcode.adapters": "Adapters translating external formats (DeepLabCut, legacy pickles, session dictionaries) into canonical contracts.",
    "fpcode.alignment": "Time-axis synchronization, sensor resampling, and monotonic frame timestamp mapping algorithms.",
    "fpcode.statistics": "Statistical estimators, circular smoothing, bootstrap resamplings, and signal summary metrics.",
    "fpcode.behavior": "Behavioral kinematics calculation, heading smoothing, and typed syllable bout segmentation.",
    "fpcode.behavior.sequences": "Behavioral transition probability matrices, chain analysis, and discrete sequence motifs.",
    "fpcode.tracking": "Multiview 3D camera correspondence, static spatial registration, and multi-camera fusion.",
    "fpcode.photometry": "Photometry preprocessing (dF/F, detrending), event window extraction, and baseline response metrics.",
    "fpcode.kpms": "Keypoint-MoSeq probabilistic generative modeling (AR-HMM/SLDS), model initialization, and training wrappers.",
    "fpcode.kpms.backend": "Performance-critical Keypoint-MoSeq execution state, memory residency, batching, and kernel dispatch.",
    "fpcode.kpms.backend.kernels": "GPU/CPU numerical kernels for continuous latent trajectories, discrete transitions, and autoregression.",
    "fpcode.viz": "Scientific visualization suite: syllable grid movies, trajectory plots, pose geometry overlays, and figures.",
    "fpcode.viz.selection": "Deterministic exemplar selection, activity density sampling, and eligibility bout filtering.",
    "fpcode.workflows": "Cohesive multi-stage pipelines: multimodal analysis, session exports, model training, and movie rendering.",
    "fpcode.cli": "Unified command-line interface entry points for training, validation, export, and diagnostics.",
    "fpcode.compat": "Isolated backward-compatibility delegation shims preserving historical pickling and notebook imports.",
}


@dataclass
class SymbolInfo:
    name: str
    kind: str  # 'class' | 'function' | 'export'
    docstring: str = ""
    signature: str = ""
    lineno: int = 1


@dataclass
class FileInfo:
    path: Path
    rel_path: str
    docstring: str = ""
    exports: list[str] = field(default_factory=list)
    classes: list[SymbolInfo] = field(default_factory=list)
    functions: list[SymbolInfo] = field(default_factory=list)
    internal_imports: set[str] = field(default_factory=set)
    external_imports: set[str] = field(default_factory=set)


@dataclass
class DirectoryInfo:
    path: Path
    rel_path: str
    module_name: str
    is_leaf: bool
    purpose: str = ""
    tags: list[str] = field(default_factory=list)
    files: list[FileInfo] = field(default_factory=list)
    subdirs: list[DirectoryInfo] = field(default_factory=list)
    internal_dependencies: set[str] = field(default_factory=set)
    external_dependencies: set[str] = field(default_factory=set)


def clean_first_line(doc: str | None) -> str:
    """Extract a concise single sentence or first line from a docstring."""
    if not doc:
        return ""
    lines = [line.strip() for line in doc.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    # Old namespace titles are not descriptions of current ownership.
    if first.startswith(("core.", "fpcode.", "syllable_generation/", "fiberphotometry/")):
        lines = [line for line in lines[1:] if not set(line) <= {"=", "-", " "}]
        first = lines[0] if lines else ""
    # Remove markdown formatting or duplicate titles
    if first.startswith(("#", "===", "---")):
        first = lines[1] if len(lines) > 1 else ""
    # Cap length neatly
    if len(first) > 140:
        first = first[:137].rsplit(" ", 1)[0] + "..."
    return first


def parse_python_file(path: Path, module_prefix: str) -> FileInfo | None:
    """Parse a Python source file using AST to extract symbols, docstrings, and imports."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except Exception as exc:
        print(f"Warning: could not parse {path}: {exc}", file=sys.stderr)
        return None

    rel_path = path.name
    docstring = clean_first_line(ast.get_docstring(tree))
    info = FileInfo(path=path, rel_path=rel_path, docstring=docstring)

    # Extract __all__ if present
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                    info.exports = [
                        elt.value for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]

    # Extract classes and functions
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            class_doc = clean_first_line(ast.get_docstring(node))
            info.classes.append(SymbolInfo(name=node.name, kind="class", docstring=class_doc, lineno=node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            func_doc = clean_first_line(ast.get_docstring(node))
            sig = f"({ast.unparse(node.args)})"
            info.functions.append(SymbolInfo(name=node.name, kind="function", docstring=func_doc,
                                             signature=sig, lineno=node.lineno))

    package = build_module_name(path.parent)
    # Include function-local optional imports; these remain static hints, not
    # proof that the dependency is initialized at import time.
    for node in ast.walk(tree):

        # Extract imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith("fpcode."):
                    info.internal_imports.add(name)
                elif "." in name:
                    base = name.split(".")[0]
                    if base in {"numpy", "scipy", "h5py", "torch", "jax", "matplotlib", "cuml", "pydantic", "pandas", "sklearn"}:
                        info.external_imports.add(base)
                elif name in {"numpy", "scipy", "h5py", "torch", "jax", "matplotlib", "cuml", "pydantic", "pandas", "sklearn"}:
                    info.external_imports.add(name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                if node.module.startswith("fpcode."):
                    info.internal_imports.add(node.module)
                elif node.level > 0:
                    # Relative import
                    try:
                        info.internal_imports.add(importlib.util.resolve_name("." * node.level + node.module, package))
                    except (ImportError, ValueError):
                        info.internal_imports.add("." * node.level + node.module)
                else:
                    base = node.module.split(".")[0]
                    if base in {"numpy", "scipy", "h5py", "torch", "jax", "matplotlib", "cuml", "pydantic", "pandas", "sklearn"}:
                        info.external_imports.add(base)
            elif node.level > 0:
                for alias in node.names:
                    try:
                        info.internal_imports.add(importlib.util.resolve_name("." * node.level + alias.name, package))
                    except (ImportError, ValueError):
                        info.internal_imports.add("." * node.level + alias.name)

    return info


def build_module_name(path: Path) -> str:
    """Derive dotted Python module name from path relative to repo root."""
    rel = path.relative_to(ROOT)
    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)


def infer_tags(dir_info: DirectoryInfo) -> list[str]:
    """Determine domain and capability tags for a directory."""
    if dir_info.module_name in CANONICAL_TAGS:
        return list(CANONICAL_TAGS[dir_info.module_name])

    # Dynamic heuristics
    tags: set[str] = set()
    dir_name = dir_info.path.name.lower()
    for token in dir_name.replace("_", "-").split("-"):
        if token and len(token) > 2 and token not in {"and", "for", "with"}:
            tags.add(token)

    # Add external dependencies
    for ext in sorted(dir_info.external_dependencies):
        tags.add(ext)

    # Add core capability keywords found in files
    for f in dir_info.files:
        for c in f.classes:
            name_lower = c.name.lower()
            if "dataset" in name_lower:
                tags.add("dataset")
            if "model" in name_lower:
                tags.add("model")
            if "pipeline" in name_lower:
                tags.add("pipeline")

    if not tags:
        tags.add(dir_name)

    return sorted(tags)[:6]


def render_leaf_map(dir_info: DirectoryInfo) -> str:
    """Generate Markdown content for a leaf directory's .folder-map.md."""
    lines: list[str] = []
    lines.append(f"# {dir_info.module_name}")
    lines.append("")
    lines.append("Generated navigation hints. Read source for behavior; the capability contract and dependency audit govern ownership.")
    lines.append("")
    lines.append(f"**Purpose:** {dir_info.purpose}")
    tag_str = ", ".join(dir_info.tags)
    lines.append(f"**Tags:** [{tag_str}]")
    lines.append("")

    # Key Exports / Symbols
    lines.append("## Declared Exports / Implementation Symbols")
    exported_names = set()
    symbol_lines: list[str] = []

    # First look at __init__.py exports
    init_file = next((f for f in dir_info.files if f.rel_path == "__init__.py"), None)
    if init_file and init_file.exports:
        for exp in sorted(init_file.exports):
            exported_names.add(exp)
            # Find matching class/func docstring in other files
            matching_doc = ""
            for f in dir_info.files:
                for c in f.classes:
                    if c.name == exp:
                        matching_doc = c.docstring
                        break
                for fn in f.functions:
                    if fn.name == exp:
                        matching_doc = fn.docstring
                        break
                if matching_doc:
                    break
            doc_suffix = f": {matching_doc}" if matching_doc else ""
            symbol_lines.append(f"- [`{exp}`](./__init__.py){doc_suffix}")

    # Also list public classes and key functions from files
    for f in dir_info.files:
        for c in f.classes:
            if c.name not in exported_names:
                doc_suffix = f": {c.docstring}" if c.docstring else ""
                symbol_lines.append(f"- [`class {c.name}`](./{f.rel_path}#L{c.lineno}){doc_suffix}")
                exported_names.add(c.name)
        for fn in f.functions:
            if fn.name not in exported_names:
                doc_suffix = f": {fn.docstring}" if fn.docstring else ""
                symbol_lines.append(f"- [`{fn.name}{fn.signature}`](./{f.rel_path}#L{fn.lineno}){doc_suffix}")
                exported_names.add(fn.name)

    if symbol_lines:
        lines.extend(symbol_lines[:20])  # Cap at top 20 to avoid prose bloat
        if len(symbol_lines) > 20:
            lines.append(f"- *(and {len(symbol_lines) - 20} additional symbols...)*")
    else:
        lines.append("- *(Internal script module without standalone public class/function exports)*")
    lines.append("")

    # Internal Dependencies
    lines.append("## Internal Dependencies")
    filtered_deps = sorted(
        dep for dep in dir_info.internal_dependencies
        if dep != dir_info.module_name and not dep.startswith(f"{dir_info.module_name}.")
    )
    if filtered_deps:
        for dep in filtered_deps:
            lines.append(f"- `{dep}`")
    else:
        lines.append("- None (Self-contained leaf module)")
    lines.append("")

    # Direct Files
    lines.append("## Direct Files")
    for f in sorted(dir_info.files, key=lambda x: x.rel_path):
        doc_part = f" - {f.docstring}" if f.docstring else ""
        lines.append(f"- [{f.rel_path}](./{f.rel_path}){doc_part}")
    lines.append("")

    return "\n".join(lines)


def render_parent_map(dir_info: DirectoryInfo) -> str:
    """Generate Markdown content for a parent directory's rollup .folder-map.md."""
    lines: list[str] = []
    lines.append(f"# {dir_info.module_name}")
    lines.append("")
    lines.append("Generated navigation hints. Import summaries include optional function-local imports and may be truncated.")
    lines.append("")
    lines.append(f"**Purpose:** {dir_info.purpose}")
    tag_str = ", ".join(dir_info.tags)
    lines.append(f"**Tags:** [{tag_str}]")
    lines.append("")

    # Sub-Modules
    lines.append("## Sub-Modules")
    for sub in sorted(dir_info.subdirs, key=lambda x: x.path.name):
        child_name = sub.path.name
        purpose = sub.purpose or "Specialized domain submodule."
        lines.append(f"- [`{child_name}/`](./{child_name}/.folder-map.md): {purpose}")
    lines.append("")

    # Direct Files in this parent folder (if any)
    if dir_info.files:
        lines.append("## Package Core Files & Entrypoints")
        for f in sorted(dir_info.files, key=lambda x: x.rel_path):
            doc_part = f" - {f.docstring}" if f.docstring else ""
            lines.append(f"- [{f.rel_path}](./{f.rel_path}){doc_part}")
        lines.append("")

    # Internal Dependencies Rollup
    lines.append("## Subsystem Dependencies")
    filtered_deps = sorted(
        dep for dep in dir_info.internal_dependencies
        if not dep.startswith(f"{dir_info.module_name}") and dep != dir_info.module_name
    )
    if filtered_deps:
        for dep in filtered_deps[:15]:
            lines.append(f"- `{dep}`")
        if len(filtered_deps) > 15:
            lines.append(f"- *(and {len(filtered_deps) - 15} further imports; inspect source/audit for the full graph)*")
    else:
        lines.append("- Self-contained foundational package")
    lines.append("")

    # Architectural Layering Note for top fpcode package
    if dir_info.module_name == "fpcode":
        lines.append("## Architectural Layering Rules")
        lines.append("The authoritative matrix is in `ops/repo/dependencies.py`; it includes cross-capability edges and named compatibility facades.")
        lines.append("See [the capability contract](../../docs/architecture/UNIT_CONTRACT.md) for responsibilities.")
        lines.append("")

    return "\n".join(lines)


def render_architecture_index(root_dirs: list[DirectoryInfo]) -> str:
    """Generate top-level ARCHITECTURE_INDEX.md at the project root."""
    lines: list[str] = []
    lines.append("# Architecture & Navigation Index")
    lines.append("")
    lines.append("Generated file/symbol maps support selective navigation. Start with [AGENTS.md](AGENTS.md),")
    lines.append("[current task state](docs/tasks/CURRENT.md) and [the capability contract](docs/architecture/UNIT_CONTRACT.md).")
    lines.append("Read the relevant map or use `rg`; there is no requirement to read every map.")
    lines.append("")

    lines.append("## 1. Subsystem Directory Topology")
    lines.append("")
    lines.append("| Directory | Role | Primary Responsibility |")
    lines.append("| :--- | :--- | :--- |")
    module_count = sum(len(d.files) for d in root_dirs if d.rel_path.startswith("src/fpcode"))
    lines.append(f"| [`src/fpcode/`](src/fpcode/.folder-map.md) | **Canonical Library** | {module_count} Python modules; implementations and installed CLI |")
    lines.append("| [`pipelines/`](pipelines/.folder-map.md) | **Historical Entry Points** | Adapters to `fpcode.cli`; preserve physical script execution |")
    lines.append("| [`ops/`](ops/.folder-map.md) | **Operations & Tooling** | Verification gates, diagnostics, repository hygiene, backups, and metrics |")
    lines.append("| [`studies/`](studies/.folder-map.md) | **Research Studies** | Domain analysis notebooks, active replication packages, and experimental units |")
    lines.append("| [`tests/`](tests/.folder-map.md) | **Test Suite** | Synthetic functional checks; explicit empirical opt-in |")
    lines.append("| [`configs/`](configs/.folder-map.md) | **Configuration** | Model hyperparameters, training profiles, environment locks, and backup policy |")
    lines.append("| [`compat/`](compat/AGENTS.md) | **Historical Namespaces** | Top-level import/file adapters; distinct from historical library implementations in `fpcode.compat` |")
    lines.append("")

    lines.append("## 2. Core Architectural Principles")
    lines.append("- The dependency audit enforces the configured matrix, not a single total ordering of capabilities.")
    lines.append("- Empirical inputs/models belong in ignored `data`, generated run products in `outputs`, or explicitly configured external storage. Transitional aliases remain.")
    lines.append("- Root `compat` preserves old namespaces; `fpcode.compat` owns historical implementations/profiles. Identity and filesystem obligations both matter.")
    lines.append("- Summaries derived from docstrings/imports are hints, not scientific policy or evidence of universal pickle compatibility.")
    lines.append("")

    lines.append("## 3. Autonomous Agent Routing Protocol")
    lines.append("When the owner is unfamiliar, use this route; direct symbol search is also appropriate:")
    lines.append("1. **Zoom 1 (Root Topology)**: Read this `ARCHITECTURE_INDEX.md` to identify candidate subsystem directories.")
    lines.append("2. **Zoom 2 (Subsystem Map)**: Open the candidate subsystem's `.folder-map.md` (e.g. `src/fpcode/.folder-map.md`) to locate the relevant subpackage.")
    lines.append("3. **Zoom 3 (Leaf Map)**: Follow a symbol/file link into source. Listed implementation symbols are not necessarily public APIs; import summaries are static hints.")
    lines.append("")

    lines.append("## 4. Key Developer Entrypoints & Gates")
    lines.append("```bash")
    lines.append("# Select an interpreter with the required dependencies; timings are environment-dependent.")
    lines.append("make test-cpu")
    lines.append("")
    lines.append("# 2. Architectural dependency verification (0 cycles, strict layering)")
    lines.append("make architecture")
    lines.append("")
    lines.append("# 3. Maintain source code lint checks")
    lines.append("make lint")
    lines.append("")
    lines.append("# 4. Verify generated maps (pre-commit and CPU CI)")
    lines.append("make folder-maps-check")
    lines.append("")
    lines.append("# 5. Regenerate all hierarchical folder maps")
    lines.append("make folder-maps")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def repository_files(target_roots: Sequence[str]) -> set[Path] | None:
    """Git-visible inputs, including uncommitted maintained files; None in exports."""
    if (ROOT / ".git").exists():
        # Include tracked and new maintained inputs, never ignored local scripts.
        # This keeps maps reproducible in a clone without staging new source.
        listing = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *target_roots],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        return {ROOT / name for name in listing.stdout.split("\0") if name}
    return None


def existing_maps(target_roots: Sequence[str]) -> set[Path]:
    """Find generated maps without following symlinks or entering excluded trees."""
    visible_files = repository_files(target_roots)
    if visible_files is not None:
        return {path for path in visible_files if path.name == ".folder-map.md"
                and not any(part in EXCLUDE_DIRS for part in path.relative_to(ROOT).parts)
                and not any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(ROOT))
                and not path.is_symlink() and path.is_file()}
    found = set()
    for target in target_roots:
        root = ROOT / target
        if root.is_symlink():
            continue
        for folder, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not (Path(folder) / d).is_symlink()]
            path = Path(folder) / ".folder-map.md"
            if ".folder-map.md" in files and not path.is_symlink():
                found.add(path)
    return found


def collect_directory_tree(target_roots: Sequence[str]) -> list[DirectoryInfo]:
    """Scan target root directories and assemble the DirectoryInfo hierarchy."""
    all_dirs: dict[Path, DirectoryInfo] = {}
    visible_files = repository_files(target_roots)
    visible_dirs = None
    if visible_files is not None:
        visible_dirs = {parent for path in visible_files if path.suffix == ".py" or
                        (path.is_relative_to(ROOT / "configs") and path.suffix in {".yaml", ".yml", ".txt"})
                        for parent in path.parents if parent.is_relative_to(ROOT)}

    for root_str in target_roots:
        root_path = ROOT / root_str
        if not root_path.exists():
            continue

        # Traverse all directories under root_path
        candidate_paths: list[Path] = []
        if root_path.is_symlink():
            continue
        for folder, dirs, _files in os.walk(root_path, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS and not (Path(folder) / d).is_symlink()
                             and (visible_dirs is None or Path(folder) / d in visible_dirs))
            candidate_paths.append(Path(folder))

        # Parse python files in each directory
        for dir_path in candidate_paths:
            py_files = sorted(dir_path.glob("*.py"))
            parsed_files: list[FileInfo] = []
            dir_internal_deps: set[str] = set()
            dir_external_deps: set[str] = set()

            for py_f in py_files:
                if py_f.is_symlink() or (visible_files is not None and py_f not in visible_files):
                    continue
                file_info = parse_python_file(py_f, dir_path.name)
                if file_info:
                    parsed_files.append(file_info)
                    dir_internal_deps.update(file_info.internal_imports)
                    dir_external_deps.update(file_info.external_imports)

            if root_str == "configs":
                for config in sorted(dir_path.iterdir()):
                    if (config.is_file() and not config.is_symlink() and config.suffix in {".yaml", ".yml", ".txt"}
                            and (visible_files is None or config in visible_files)):
                        parsed_files.append(FileInfo(path=config, rel_path=config.name, docstring="Configuration/profile; inspect its schema and consumer before execution."))

            module_name = build_module_name(dir_path)

            # Determine purpose
            init_file = next((f for f in parsed_files if f.rel_path == "__init__.py"), None)
            purpose = ""
            if module_name in CANONICAL_PURPOSES:
                purpose = CANONICAL_PURPOSES[module_name]
            elif init_file and init_file.docstring:
                purpose = init_file.docstring
            elif parsed_files:
                # Use first file docstring with content
                for f in parsed_files:
                    if f.docstring and not f.docstring.startswith("fpcode"):
                        purpose = f.docstring
                        break

            if not purpose:
                purpose = f"Module and utility implementations for {dir_path.name}."

            dir_info = DirectoryInfo(
                path=dir_path,
                rel_path=str(dir_path.relative_to(ROOT)),
                module_name=module_name,
                is_leaf=True,  # Will update below
                purpose=purpose,
                files=parsed_files,
                internal_dependencies=dir_internal_deps,
                external_dependencies=dir_external_deps,
            )
            dir_info.tags = infer_tags(dir_info)
            all_dirs[dir_path] = dir_info

    # Establish parent-child relationships and rollup dependencies
    for dir_path, dir_info in sorted(all_dirs.items(), key=lambda item: len(item[0].parts), reverse=True):
        # Find immediate child directories in all_dirs
        child_dirs = [
            child for p, child in all_dirs.items()
            if p.parent == dir_path
        ]
        if not dir_info.files and not child_dirs:
            del all_dirs[dir_path]
            continue
        if child_dirs:
            dir_info.is_leaf = False
            dir_info.subdirs = child_dirs
            # Roll up dependencies from children
            for child in child_dirs:
                dir_info.internal_dependencies.update(child.internal_dependencies)
                dir_info.external_dependencies.update(child.external_dependencies)

    return list(all_dirs.values())


def execute(
    target_roots: Sequence[str] = DEFAULT_TARGETS,
    check_only: bool = False,
    show_diff: bool = False,
    quiet: bool = False,
) -> int:
    """Generate or check .folder-map.md and ARCHITECTURE_INDEX.md."""
    dir_infos = collect_directory_tree(target_roots)
    discrepancies: list[str] = []
    generated_count = 0
    expected_maps = {d.path / ".folder-map.md" for d in dir_infos}
    for stale in sorted(existing_maps(target_roots) - expected_maps):
        if check_only:
            discrepancies.append(str(stale.relative_to(ROOT)) + " (obsolete)")
        else:
            stale.unlink()
            if not quiet:
                print(f"  [removed obsolete map] {stale.relative_to(ROOT)}")

    # Sort bottom-up: deepest directories first
    sorted_dirs = sorted(dir_infos, key=lambda d: len(d.path.parts), reverse=True)

    for d in sorted_dirs:
        # Only create .folder-map.md if directory has files or subdirs with maps
        if not d.files and not d.subdirs:
            continue

        target_file = d.path / ".folder-map.md"
        content = render_leaf_map(d) if d.is_leaf else render_parent_map(d)

        existing = target_file.read_text(encoding="utf-8") if target_file.exists() else None

        if existing != content:
            rel_target = target_file.relative_to(ROOT)
            if check_only:
                discrepancies.append(str(rel_target))
                if show_diff and existing is not None:
                    diff = difflib.unified_diff(
                        existing.splitlines(keepends=True),
                        content.splitlines(keepends=True),
                        fromfile=f"a/{rel_target}",
                        tofile=f"b/{rel_target}",
                    )
                    sys.stdout.writelines(diff)
            else:
                target_file.write_text(content, encoding="utf-8")
                generated_count += 1
                if not quiet:
                    print(f"  [written] {rel_target}")

    # Generate or check root ARCHITECTURE_INDEX.md if src/fpcode is among targets
    if any("src/fpcode" in str(r) for r in target_roots):
        arch_file = ROOT / "ARCHITECTURE_INDEX.md"
        arch_content = render_architecture_index(dir_infos)
        existing_arch = arch_file.read_text(encoding="utf-8") if arch_file.exists() else None

        if existing_arch != arch_content:
            if check_only:
                discrepancies.append("ARCHITECTURE_INDEX.md")
                if show_diff and existing_arch is not None:
                    diff = difflib.unified_diff(
                        existing_arch.splitlines(keepends=True),
                        arch_content.splitlines(keepends=True),
                        fromfile="a/ARCHITECTURE_INDEX.md",
                        tofile="b/ARCHITECTURE_INDEX.md",
                    )
                    sys.stdout.writelines(diff)
            else:
                arch_file.write_text(arch_content, encoding="utf-8")
                generated_count += 1
                if not quiet:
                    print("  [written] ARCHITECTURE_INDEX.md")

    if check_only:
        if discrepancies:
            print(f"Error: {len(discrepancies)} documentation maps are out of date:", file=sys.stderr)
            for path in discrepancies:
                print(f"  - {path}", file=sys.stderr)
            print("Run 'make folder-maps' or 'python -B ops/repo/generate_folder_maps.py' to synchronize.", file=sys.stderr)
            return 1
        else:
            if not quiet:
                print(f"All folder maps and architecture index are synchronized ({len(dir_infos)} directories verified).")
            return 0

    if not quiet:
        print(f"Successfully generated/updated {generated_count} folder maps.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="Target directories relative to repository root to index (default: %(default)s)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify folder maps are up-to-date without modifying files (exits 1 on drift)",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Display unified diff when checking or updating",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output",
    )
    args = parser.parse_args(argv)
    return execute(
        target_roots=args.targets,
        check_only=args.check,
        show_diff=args.diff,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    raise SystemExit(main())
