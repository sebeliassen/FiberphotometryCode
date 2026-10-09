"""
scripts.verify_tier_a_backup
============================
Rigorous content hash verification comparing live repository Tier-A artifacts
against their physical backup mirror on /mnt/hdd/backups/TierA_backup_20261009.
Consults MOVES.tsv to resolve relocated paths.
"""

import csv
import hashlib
import os
import sys
import time
from pathlib import Path

BACKUP_ROOT = Path(os.environ.get("TIER_A_BACKUP_DIR", "/mnt/hdd/backups/TierA_backup_20261009"))
REPO_ROOT = Path(__file__).resolve().parent.parent
MOVES_PATH = REPO_ROOT / "MOVES.tsv"


def compute_content_sha256(path: Path) -> str:
    """Computes exact SHA-256 over file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_moves_map() -> dict:
    moves = {}
    if MOVES_PATH.exists():
        with open(MOVES_PATH, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                moves[row["source_path"]] = row["target_path"]
    return moves


def main():
    if not BACKUP_ROOT.exists():
        print(f"[ERROR] Backup directory not found: {BACKUP_ROOT}")
        sys.exit(1)

    print("=" * 80)
    print("VERIFYING TIER-A BACKUP CONTENT HASHES (LIVE VS BACKUP)")
    print(f"Source Repository : {REPO_ROOT}")
    print(f"Backup Destination: {BACKUP_ROOT}")
    print("=" * 80)

    moves_map = load_moves_map()
    checked = 0
    matched = 0
    symlinks_verified = 0
    mismatched = []
    start_time = time.time()

    # Discover all backed-up files on /mnt/hdd
    backup_files = []
    for root, _, files in os.walk(BACKUP_ROOT):
        for f in files:
            full_backup_path = Path(root) / f
            rel = full_backup_path.relative_to(BACKUP_ROOT)
            backup_files.append((rel, full_backup_path))

    print(f"Found {len(backup_files)} physical backup files on /mnt/hdd. Verifying...")

    for rel_path, backup_path in backup_files:
        rel_str = str(rel_path)
        live_path = REPO_ROOT / rel_path

        # Resolve live path if moved per MOVES.tsv
        if not live_path.exists() and not live_path.is_symlink():
            for src_pref, tgt_pref in moves_map.items():
                if rel_str == src_pref or rel_str.startswith(src_pref + "/"):
                    new_rel = rel_str.replace(src_pref, tgt_pref, 1)
                    cand = REPO_ROOT / new_rel
                    if cand.exists() or cand.is_symlink():
                        live_path = cand
                        break

        # If backup is a symlink, verify symlink target
        if backup_path.is_symlink():
            if live_path.is_symlink():
                t_b = os.readlink(backup_path)
                t_l = os.readlink(live_path)
                if t_b == t_l:
                    symlinks_verified += 1
                    checked += 1
                else:
                    mismatched.append((rel_str, f"Symlink target mismatch: {t_b} != {t_l}"))
            else:
                mismatched.append((rel_str, "Backup is symlink but live file is not"))
            continue

        if not live_path.exists():
            mismatched.append((rel_str, f"Live file missing from repository (checked {live_path})"))
            continue

        checked += 1
        st_backup = backup_path.lstat()
        st_live = live_path.lstat()

        if st_backup.st_size != st_live.st_size:
            mismatched.append((rel_str, f"File size mismatch: backup={st_backup.st_size} != live={st_live.st_size}"))
            continue

        # Compare cryptographic hashes
        h_backup = compute_content_sha256(backup_path)
        h_live = compute_content_sha256(live_path)

        if h_backup == h_live:
            matched += 1
        else:
            mismatched.append((rel_str, f"Hash mismatch: backup={h_backup[:16]} != live={h_live[:16]}"))

        if (matched + symlinks_verified) % 100 == 0:
            print(f"  Verified {matched + symlinks_verified}/{len(backup_files)} files...")

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"Total Tier-A backup files inspected : {checked}")
    print(f"Exact content hash matches verified : {matched}")
    print(f"Symlinks verified identical         : {symlinks_verified}")
    print(f"Discrepancies / Mismatches          : {len(mismatched)}")
    print(f"Verification Duration               : {elapsed:.2f}s")
    print("=" * 80)

    if mismatched:
        print("[FAILED] Integrity check encountered discrepancies!")
        for path, err in mismatched[:10]:
            print(f"  Mismatched: {path} -> {err}")
        sys.exit(1)
    else:
        print("🎉 [SUCCESS] 100% of physical Tier-A backup files match live originals byte-for-byte!")


if __name__ == "__main__":
    main()
