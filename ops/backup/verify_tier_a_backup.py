"""Legacy mirror comparison with expected coverage and full live-vs-backup hashes.

This command reads every eligible Tier-A file; use coverage.py for bounded spot
checks. New snapshots use snapshot.py verify, independently of the changing live
tree. Partial ht16m inventory fingerprints are not treated as complete hashes.
"""

import csv
import hashlib
import os
import sys
import time
from pathlib import Path

if __package__:
    from .coverage import backup_entries, coverage_report, load_expected, load_moves, resolve_live_path
else:
    from coverage import backup_entries, coverage_report, load_expected, load_moves, resolve_live_path

BACKUP_ROOT = Path(os.environ.get("TIER_A_BACKUP_DIR", "/mnt/hdd/backups/TierA_backup_20261009"))
REPO_ROOT = Path(os.environ.get("TIER_A_SOURCE_DIR", str(Path(__file__).resolve().parents[2])))
MOVES_PATH = Path(os.environ.get("TIER_A_MOVES", str(REPO_ROOT / "registry" / "MOVES.tsv")))
INVENTORY_PATH = Path(os.environ.get("TIER_A_INVENTORY", str(REPO_ROOT / "registry" / "ARTIFACT_INVENTORY.tsv")))


def compute_content_sha256(path: Path) -> str:
    """Computes exact SHA-256 over file contents."""
    h = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as f:
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

    try:
        coverage = coverage_report(BACKUP_ROOT, INVENTORY_PATH, REPO_ROOT, MOVES_PATH, spot_check_count=0)
    except (OSError, ValueError, KeyError) as error:
        print(f"[ERROR] Expected backup coverage could not be verified: {error}")
        sys.exit(1)
    if coverage["state"] != "coverage_checked":
        print(f"[ERROR] Incomplete legacy mirror: missing={len(coverage['missing'])}, "
              f"unexpected={len(coverage['unexpected'])}, mismatched={len(coverage['metadata_mismatches'])}")
        sys.exit(1)

    print("=" * 80)
    print("VERIFYING TIER-A BACKUP CONTENT HASHES (LIVE VS BACKUP)")
    print(f"Source Repository : {REPO_ROOT}")
    print(f"Backup Destination: {BACKUP_ROOT}")
    print("=" * 80)

    moves = load_moves(MOVES_PATH)
    checked = 0
    matched = 0
    symlinks_verified = 0
    mismatched = []
    start_time = time.time()

    # Discover all backed-up files on /mnt/hdd
    expected = load_expected(INVENTORY_PATH)
    backup_files = [(Path(relative), path) for relative, path in backup_entries(BACKUP_ROOT).items() if relative in expected]

    print(f"Found {len(backup_files)} physical backup files on /mnt/hdd. Verifying...")

    for rel_path, backup_path in backup_files:
        rel_str = str(rel_path)
        live_path = REPO_ROOT / rel_path

        # Resolve relocated regular files and compatibility symlinks inside this repository.
        if not backup_path.is_symlink():
            resolved = resolve_live_path(REPO_ROOT, rel_str, moves)
            if resolved is not None:
                live_path = resolved

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
        print("[SUCCESS] Expected Tier-A coverage is complete and backed-up files match current live originals.")


if __name__ == "__main__":
    main()
