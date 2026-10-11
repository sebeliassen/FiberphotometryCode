#!/usr/bin/env python3
"""
Historical October 9 migration utility. Its source namespaces refer to the
pre-migration tree; it is retained as a record, not the current cleanup workflow.

ops/repo/quarantine_class_c.py

Executes Step 8.1 of the Architectural Refactoring:
1. Diffs dated backups against active canonical copies and writes registry/BACKUP_DIFFS.tsv.
2. Moves all Class C legacy and backup files/directories to archive/legacy/.
3. Updates registry/MOVES.tsv with every quarantined path.
"""

import filecmp
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

BACKUP_PAIRS = [
    ("syllable_generation/analysis/ulrik_kinematics_backup_20260526", "syllable_generation/analysis/ulrik_kinematics"),
    ("syllable_generation/analysis/ulrik_kinematics_backup_20260527", "syllable_generation/analysis/ulrik_kinematics"),
    ("syllable_generation/analysis/language_models_legacy_backup", "syllable_generation/analysis/language_models"),
    ("syllable_generation/analysis/dre_bk", "syllable_generation/analysis/deep_research_experiments"),
]

CODE_EXTS = {".py", ".sh", ".yml", ".yaml", ".md"}
SKIP_DIRS = {"wandb", "output", "cache", "__pycache__"}

def audit_backup_diffs():
    print("Auditing dated backups against canonical versions (code files)...")
    diff_records = []

    for backup_dir_rel, canon_dir_rel in BACKUP_PAIRS:
        bp = REPO_ROOT / backup_dir_rel
        cp = REPO_ROOT / canon_dir_rel
        if not bp.exists():
            continue

        for f in bp.rglob("*"):
            if any(part in SKIP_DIRS for part in f.parts):
                continue
            if f.is_file() and f.suffix in CODE_EXTS:
                rel = f.relative_to(bp)
                canon_file = cp / rel
                backup_rel = f.relative_to(REPO_ROOT)
                canon_rel = canon_file.relative_to(REPO_ROOT) if canon_file.exists() else Path("NONE")

                if not canon_file.exists():
                    status = "ONLY_IN_BACKUP"
                    c_size = 0
                elif filecmp.cmp(f, canon_file, shallow=False):
                    status = "IDENTICAL"
                    c_size = canon_file.stat().st_size
                else:
                    status = "DIVERGED"
                    c_size = canon_file.stat().st_size

                diff_records.append((str(backup_rel), str(canon_rel), status, f.stat().st_size, c_size))

    diff_tsv = REPO_ROOT / "registry" / "BACKUP_DIFFS.tsv"
    with open(diff_tsv, "w") as f:
        f.write("BackupPath\tCanonicalPath\tStatus\tBackupSizeBytes\tCanonicalSizeBytes\n")
        for rec in diff_records:
            f.write(f"{rec[0]}\t{rec[1]}\t{rec[2]}\t{rec[3]}\t{rec[4]}\n")

    print(f"Recorded {len(diff_records)} file diff comparisons to {diff_tsv.relative_to(REPO_ROOT)}")
    return diff_records

def execute_quarantine():
    os.chdir(REPO_ROOT)
    triage_tsv = REPO_ROOT / "registry" / "TRIAGE.tsv"
    with open(triage_tsv) as f:
        lines = f.read().splitlines()

    class_c_files = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 11 and parts[9] == "C":
            class_c_files.append((parts[0], int(parts[1]), parts[6], parts[10]))

    print(f"Total Class C files to quarantine: {len(class_c_files)}")

    # Distinct bulk directories to move
    bulk_legacy_dirs = [
        "syllable_generation/analysis/ulrik_kinematics_backup_20260526",
        "syllable_generation/analysis/ulrik_kinematics_backup_20260527",
        "syllable_generation/analysis/language_models_legacy_backup",
        "syllable_generation/analysis/dre_bk",
        "syllable_generation/multiview_top_bot_experiments/archive_prototypes",
        "syllable_generation/multiview_top_bot_experiments/scratch_expert_package",
        "fiberphotometry/legacy",
        "syllable_generation/analysis/language_models/legacy_sweeps",
    ]

    moved_records = []

    # 1. Move bulk legacy directories
    for bdir_rel in bulk_legacy_dirs:
        src = REPO_ROOT / bdir_rel
        if src.exists():
            dst = REPO_ROOT / "archive" / "legacy" / bdir_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            print(f"Moved directory: {bdir_rel} -> {dst.relative_to(REPO_ROOT)}")
            moved_records.append((bdir_rel, str(dst.relative_to(REPO_ROOT)), "QUARANTINE_LEGACY_DIR", "Bulk legacy / backup directory quarantine", "NO"))

    # 2. Move individual files that remain
    for p_rel, line_cnt, dup_of, dest_rel in class_c_files:
        src = REPO_ROOT / p_rel
        # If it was already moved as part of bulk dir, skip
        if any(p_rel.startswith(bdir) for bdir in bulk_legacy_dirs):
            continue

        if src.exists():
            dst = REPO_ROOT / dest_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"Moved file: {p_rel} -> {dest_rel}")
            moved_records.append((p_rel, dest_rel, "QUARANTINE_LEGACY_FILE", f"Class C duplicate / legacy (duplicate of {dup_of})", "NO"))

    # 3. Append to registry/MOVES.tsv
    moves_tsv = REPO_ROOT / "registry" / "MOVES.tsv"
    with open(moves_tsv, "a") as f:
        for r in moved_records:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\n")

    print(f"Quarantined {len(moved_records)} units to archive/legacy/ and updated registry/MOVES.tsv.")

if __name__ == "__main__":
    audit_backup_diffs()
    execute_quarantine()
