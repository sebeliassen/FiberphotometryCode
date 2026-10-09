#!/usr/bin/env python3
"""
ops/repo/build_triage.py

Automated AST and Dependency Graph Triage for FiberphotometryCode Analysis Lump.
Generates registry/TRIAGE.tsv classifying candidate files into:
- Class A: Reusable library code -> src/fpcode/
- Class B: Study-specific scripts/drivers -> studies/<study>/
- Class C: Legacy/backup/duplicate -> archive/legacy/
- Vendor: Third-party fork -> vendor/
"""

import ast
import hashlib
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]

EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "archive",
    "tests",
    "ops",
    "pipelines",
    "src",
    "core",
    "outputs",
    "scratch",
    "reports",
    "logs",
}

def compute_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    try:
        hasher.update(filepath.read_bytes())
        return hasher.hexdigest()
    except Exception:
        return ""

def analyze_ast(filepath: Path):
    try:
        content = filepath.read_text(errors="ignore")
        tree = ast.parse(content, filename=str(filepath))
    except Exception:
        return False, [], [], set(), content if "content" in locals() else ""

    has_main = False
    has_argparse = False
    classes = []
    functions = []
    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Check for if __name__ == '__main__':
            if isinstance(node.test, ast.Compare):
                left = node.test.left
                if isinstance(left, ast.Name) and left.id == "__name__":
                    has_main = True
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "ArgumentParser":
                has_argparse = True
            elif isinstance(node.func, ast.Name) and node.func.id == "ArgumentParser":
                has_argparse = True
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)

    return (has_main or has_argparse), classes, functions, imported_modules, content

def get_proposed_destination(rel_path: str, classification: str) -> str:
    parts = Path(rel_path).parts
    filename = parts[-1]

    if classification == "C":
        return f"archive/legacy/{rel_path}"

    if classification == "VENDOR":
        # syllable_generation/syllabizers/keypoint_moseq_syll/... -> vendor/keypoint_moseq_syll/...
        if "keypoint_moseq_syll" in rel_path:
            sub = rel_path.split("keypoint_moseq_syll", 1)[1].lstrip("/")
            return f"vendor/keypoint_moseq_syll/{sub}"
        return f"vendor/{rel_path}"

    if classification == "A":
        # Library destinations in src/fpcode/
        if "language_models" in rel_path:
            return f"src/fpcode/behavior/language_models/{filename}"
        elif "sequence_viz" in rel_path or "syllable_chains" in rel_path:
            return f"src/fpcode/viz/sequences/{filename}"
        elif "ulrik_kinematics" in rel_path:
            return f"src/fpcode/behavior/kinematics/{filename}"
        elif "loaders" in rel_path:
            return f"src/fpcode/common/io/{filename}"
        elif "videogen" in rel_path:
            return f"src/fpcode/viz/videogen/{filename}"
        elif "diagnostics" in rel_path:
            return f"ops/diagnostics/{filename}"
        elif "fiberphotometry" in rel_path:
            return f"src/fpcode/photometry/{filename}"
        elif filename == "datamodel.py":
            return f"src/fpcode/behavior/data.py"
        elif "shmoseq" in rel_path:
            return f"src/fpcode/behavior/shmoseq/{filename}"
        elif "replications" in rel_path:
            return f"src/fpcode/behavior/replications/{filename}"
        elif "multiview_top_bot_experiments" in rel_path:
            return f"src/fpcode/kpms/{filename}"
        else:
            return f"src/fpcode/behavior/{filename}"

    # Class B: Study destinations in studies/<domain>/<study>/
    if "language_models" in rel_path:
        sub = rel_path.split("language_models", 1)[1].lstrip("/")
        return f"studies/behavior/language_models/{sub}"
    elif "sequence_viz" in rel_path:
        sub = rel_path.split("sequence_viz", 1)[1].lstrip("/")
        return f"studies/behavior/sequence_viz/{sub}"
    elif "ulrik_kinematics" in rel_path:
        sub = rel_path.split("ulrik_kinematics", 1)[1].lstrip("/")
        return f"studies/behavior/kinematics_ulrik/{sub}"
    elif "replications" in rel_path:
        sub = rel_path.split("replications", 1)[1].lstrip("/")
        return f"studies/behavior/replications_datta/{sub}"
    elif "shmoseq" in rel_path:
        sub = rel_path.split("shmoseq", 1)[1].lstrip("/")
        return f"studies/behavior/shmoseq/{sub}"
    elif "deep_research_experiments" in rel_path:
        sub = rel_path.split("deep_research_experiments", 1)[1].lstrip("/")
        return f"studies/behavior/deep_research_experiments/{sub}"
    elif "karen" in rel_path:
        return f"studies/behavior/karen_studies/{filename}"
    elif "turnlab" in rel_path:
        sub = rel_path.split("turnlab", 1)[1].lstrip("/")
        return f"studies/photometry/turnlab/{sub}"
    elif "monoamine_interactions" in rel_path:
        sub = rel_path.split("monoamine_interactions", 1)[1].lstrip("/")
        return f"studies/photometry/monoamine_interactions/{sub}"
    elif "fiberphotometry" in rel_path:
        sub = rel_path.split("fiberphotometry", 1)[1].lstrip("/")
        return f"studies/photometry/exploratory/{sub}"
    elif "multiview_top_bot_experiments" in rel_path:
        sub = rel_path.split("multiview_top_bot_experiments", 1)[1].lstrip("/")
        return f"studies/behavior/multiview_experiments/{sub}"
    elif "diagnostics" in rel_path:
        return f"ops/diagnostics/{filename}"
    elif "videogen" in rel_path:
        return f"studies/behavior/videogen/{filename}"
    elif "loaders" in rel_path:
        return f"studies/behavior/loaders/{filename}"
    elif "projects" in rel_path:
        return f"studies/behavior/projects/{filename}"
    elif parts[0] == "scripts" or filename == "update_nb_cell.py":
        return f"ops/scripts/{filename}"
    return f"studies/behavior/misc/{rel_path}"

def main():
    os.chdir(REPO_ROOT)
    print("Collecting files for triage analysis...")

    # Identify all candidate files
    candidate_paths = []
    all_repo_py_files = []
    all_repo_text_files = []

    for p in REPO_ROOT.rglob("*"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.is_file():
            if p.suffix == ".py":
                all_repo_py_files.append(p)
                # Scope check
                rel = str(p.relative_to(REPO_ROOT))
                if (
                    rel.startswith("syllable_generation")
                    or rel.startswith("fiberphotometry")
                    or rel.startswith("monoamine_interactions")
                    or rel.startswith("studies/monoamine_interactions")
                    or rel.startswith("scripts")
                    or "/" not in rel  # top-level scripts
                ):
                    candidate_paths.append(p)
            if p.suffix in (".py", ".sh", ".ipynb", ".yml", ".yaml"):
                all_repo_text_files.append(p)

    candidate_paths = sorted(list(set(candidate_paths)))
    print(f"Found {len(candidate_paths)} candidate Python files in scope.")

    # 1. Compute Hashes & Detect Duplicates
    hash_to_files = defaultdict(list)
    file_hashes = {}
    file_info = {}

    for p in candidate_paths:
        rel = str(p.relative_to(REPO_ROOT))
        h = compute_md5(p)
        file_hashes[rel] = h
        hash_to_files[h].append(rel)

    # 2. Parse AST for each candidate
    print("Analyzing ASTs...")
    module_to_path = {}
    path_to_ast = {}
    for p in candidate_paths:
        rel = str(p.relative_to(REPO_ROOT))
        has_main, classes, funcs, imports, content = analyze_ast(p)
        stem = p.stem
        module_to_path[stem] = rel
        lines = len(content.splitlines())
        mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")
        path_to_ast[rel] = {
            "has_main": has_main,
            "classes": classes,
            "funcs": funcs,
            "imports": imports,
            "lines": lines,
            "mtime": mtime,
            "content": content,
            "stem": stem,
        }

    # 3. Compute In-Degree (imported by) & Out-Degree
    print("Computing import graph & reference counts...")
    in_degree = defaultdict(set)
    out_degree = defaultdict(set)

    for rel, info in path_to_ast.items():
        for imp in info["imports"]:
            imp_base = imp.split(".")[0]
            imp_tail = imp.split(".")[-1]
            # check if imp matches another candidate module
            for cand_rel, cand_info in path_to_ast.items():
                if cand_rel == rel:
                    continue
                if cand_info["stem"] in (imp_base, imp_tail):
                    in_degree[cand_rel].add(rel)
                    out_degree[rel].add(cand_rel)

    # 4. Search References across all scripts/notebooks/py
    ref_counts = defaultdict(int)
    print("Searching text references across repo...")
    file_text_cache = {}
    for tf in all_repo_text_files:
        try:
            file_text_cache[tf] = tf.read_text(errors="ignore")
        except Exception:
            pass

    for p in candidate_paths:
        rel = str(p.relative_to(REPO_ROOT))
        stem = p.stem
        count = 0
        for tf, txt in file_text_cache.items():
            if tf == p:
                continue
            if stem in txt:
                count += 1
        ref_counts[rel] = count

    # 5. Classify each file
    print("Classifying files...")
    rows = []
    summary_counts = defaultdict(lambda: defaultdict(int))
    summary_lines = defaultdict(lambda: defaultdict(int))

    for p in candidate_paths:
        rel = str(p.relative_to(REPO_ROOT))
        ast_data = path_to_ast[rel]
        h = file_hashes[rel]
        lines = ast_data["lines"]
        in_deg = len(in_degree[rel])
        out_deg = len(out_degree[rel])
        has_main = ast_data["has_main"]
        defs_summary = f"{len(ast_data['classes'])} classes, {len(ast_data['funcs'])} funcs"
        refs = ref_counts[rel]
        mtime = ast_data["mtime"]

        # Duplicate check (ignore 0-byte or empty __init__.py files)
        dup_of = ""
        same_hashes = hash_to_files[h]
        if lines > 0 and not rel.endswith("__init__.py") and len(same_hashes) > 1 and same_hashes[0] != rel:
            dup_of = same_hashes[0]

        # Classification rules:
        # Check Class C first:
        is_backup = (
            "backup" in rel.lower()
            or "legacy" in rel.lower()
            or "dre_bk" in rel.lower()
            or "archive_prototypes" in rel.lower()
            or "scratch_expert_package" in rel.lower()
            or (dup_of != "" and not rel.endswith("__init__.py"))
        )

        is_vendor = "keypoint_moseq_syll" in rel

        if is_vendor:
            cls = "VENDOR"
        elif is_backup:
            cls = "C"
        elif in_deg >= 2 or (
            not has_main
            and (len(ast_data["classes"]) > 0 or len(ast_data["funcs"]) >= 3)
            and any(k in rel for k in ["loaders", "videogen", "diagnostics", "processing", "metrics", "data", "datamodel"])
        ):
            cls = "A"
        else:
            cls = "B"

        dest = get_proposed_destination(rel, cls)
        rows.append((rel, lines, in_deg, out_deg, has_main, defs_summary, dup_of, refs, mtime, cls, dest))

        # Folder bucket for summary
        top_bucket = Path(rel).parts[0]
        if len(Path(rel).parts) > 2 and top_bucket == "syllable_generation":
            top_bucket = f"{Path(rel).parts[0]}/{Path(rel).parts[1]}"
            if len(Path(rel).parts) > 3 and Path(rel).parts[1] == "analysis":
                top_bucket = f"{Path(rel).parts[0]}/{Path(rel).parts[1]}/{Path(rel).parts[2]}"
        summary_counts[top_bucket][cls] += 1
        summary_lines[top_bucket][cls] += lines

    # Write registry/TRIAGE.tsv
    triage_path = REPO_ROOT / "registry" / "TRIAGE.tsv"
    triage_path.parent.mkdir(parents=True, exist_ok=True)
    with open(triage_path, "w") as f:
        f.write("Path\tLines\tInDegree\tOutDegree\tHasMainOrArgparse\tDefinitions\tDuplicateOf\tReferencesCount\tLastModified\tClass\tProposedDestination\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]}\t{r[6]}\t{r[7]}\t{r[8]}\t{r[9]}\t{r[10]}\n")

    print(f"\nWrote {len(rows)} entries to {triage_path.relative_to(REPO_ROOT)}")

    # Print Summary Table
    print("\n" + "=" * 90)
    print(f"{'Folder / Domain Area':<48} | {'Class A':<8} | {'Class B':<8} | {'Class C':<8} | {'Vendor':<8} | {'Total Files':<10}")
    print("-" * 90)
    total_a = total_b = total_c = total_v = 0
    total_lines_a = total_lines_b = total_lines_c = total_lines_v = 0

    for bucket in sorted(summary_counts.keys()):
        ca = summary_counts[bucket]["A"]
        cb = summary_counts[bucket]["B"]
        cc = summary_counts[bucket]["C"]
        cv = summary_counts[bucket]["VENDOR"]
        tot = ca + cb + cc + cv
        total_a += ca
        total_b += cb
        total_c += cc
        total_v += cv
        total_lines_a += summary_lines[bucket]["A"]
        total_lines_b += summary_lines[bucket]["B"]
        total_lines_c += summary_lines[bucket]["C"]
        total_lines_v += summary_lines[bucket]["VENDOR"]
        print(f"{bucket:<48} | {ca:<8} | {cb:<8} | {cc:<8} | {cv:<8} | {tot:<10}")

    print("=" * 90)
    total_all = total_a + total_b + total_c + total_v
    print(f"{'TOTAL FILES':<48} | {total_a:<8} | {total_b:<8} | {total_c:<8} | {total_v:<8} | {total_all:<10}")
    print(f"{'TOTAL LINES':<48} | {total_lines_a:<8} | {total_lines_b:<8} | {total_lines_c:<8} | {total_lines_v:<8} | {total_lines_a+total_lines_b+total_lines_c+total_lines_v:<10}")
    print("=" * 90)

if __name__ == "__main__":
    main()
