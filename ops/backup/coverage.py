"""Read-only coverage and bounded spot checks for the existing Tier-A mirror.

This is not a full integrity verification: inventory ht16m values are partial
fingerprints, and only explicitly selected small files receive full SHA-256.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKUP_ROOT = Path("/mnt/hdd/backups/TierA_backup_20261009")
# Phase-0 provenance was copied beside the Tier-A payload, outside its inventory.
AUXILIARY_METADATA = frozenset({"ARTIFACT_INVENTORY.tsv", "BACKUP_REPORT.md",
                               "PERMISSIONS_MODIFIED_TIER_A.tsv", "restore_tier_a_permissions.sh"})


def _safe_path(value):
    parsed = PurePosixPath(value)
    if not value or parsed.is_absolute() or ".." in parsed.parts or value != parsed.as_posix() or value == ".":
        raise ValueError(f"Unsafe inventory path: {value!r}")
    return value


def load_moves(path):
    moves = []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            moves.append((_safe_path(row["source_path"]), _safe_path(row["target_path"])))
    return sorted(moves, key=lambda item: len(item[0]), reverse=True)


def resolve_live_path(root, relative, moves):
    """Handle moved files, moved parent directories and compatibility symlinks."""
    root = Path(root).resolve()
    candidates, visited = [relative], set()
    while candidates:
        candidate = candidates.pop(0)
        if candidate in visited:
            continue
        visited.add(candidate)
        path = root / candidate
        if os.path.lexists(path):
            if path.is_symlink():
                try:
                    resolved = path.resolve(strict=True)
                except (OSError, RuntimeError):
                    resolved = None
                if resolved is not None and (root == resolved or root in resolved.parents):
                    return resolved
            else:
                resolved = path.resolve(strict=True)
                if root == resolved or root in resolved.parents:
                    return resolved
        for source, target in moves:
            if candidate == source or candidate.startswith(source + "/"):
                mapped = target + candidate[len(source):]
                if mapped not in visited:
                    candidates.append(mapped)
        if len(visited) > 100:
            raise ValueError(f"Excessively complex moves for {relative}")
    return None


def load_expected(inventory, max_file_bytes=2 * 1024**3):
    expected = {}
    with Path(inventory).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["tier"] != "A" or int(row["size"]) >= max_file_bytes:
                continue
            relative = _safe_path(row["path"])
            if relative in expected:
                raise ValueError(f"Duplicate inventory path: {relative}")
            expected[relative] = {"size": int(row["size"]), "hash": row["hash"]}
    if not expected:
        raise ValueError("Expected Tier-A inventory is empty")
    return expected


def backup_entries(root):
    """Enumerate file and directory links without walking their targets."""
    found = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        links = [name for name in dirs if (Path(directory) / name).is_symlink()]
        dirs[:] = [name for name in dirs if name not in links]
        for name in files + links:
            path = Path(directory) / name
            found[path.relative_to(root).as_posix()] = path
    return found


def _bounded_hash(path, limit):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"Spot-check limit/type violation: {path}")
        content = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
        if (len(content) > limit or len(content) != info.st_size
                or (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                != (info.st_size, info.st_mtime_ns, info.st_ctime_ns)):
            raise ValueError(f"Spot-check file changed: {path}")
    return hashlib.sha256(content).hexdigest()


def coverage_report(backup_root, inventory, source_root, moves_path, max_file_bytes=2 * 1024**3,
                    spot_check_count=10, spot_check_max_bytes=1024**2):
    backup_root, source_root = Path(backup_root).resolve(), Path(source_root).resolve()
    if not backup_root.is_dir():
        raise ValueError("Existing backup directory is absent")
    if spot_check_count < 0 or spot_check_max_bytes < 1:
        raise ValueError("Spot-check count and byte limit must be valid")
    expected = load_expected(inventory, max_file_bytes)
    found = backup_entries(backup_root)
    if not found:
        raise ValueError("Existing backup is empty")
    moves = load_moves(moves_path)
    missing = sorted(set(expected) - set(found))
    extras = set(found) - set(expected)
    auxiliary = sorted(path for path in extras if path in AUXILIARY_METADATA
                       and stat.S_ISREG(found[path].lstat().st_mode))
    unexpected = sorted(extras - set(auxiliary))
    mismatches, matched, reuse_candidates = [], [], []
    for relative in sorted(set(expected) & set(found)):
        row, path = expected[relative], found[relative]
        info = path.lstat()
        is_link = row["hash"].startswith("symlink:")
        if is_link:
            if not stat.S_ISLNK(info.st_mode) or os.readlink(path) != row["hash"][len("symlink:"):]:
                mismatches.append({"path": relative, "reason": "symlink_type_or_target"})
                continue
        elif not stat.S_ISREG(info.st_mode) or info.st_size != row["size"]:
            mismatches.append({"path": relative, "reason": "file_type_or_size"})
            continue
        matched.append(relative)
        live = resolve_live_path(source_root, relative, moves)
        if live is not None and live.is_file() and not is_link and live.stat().st_size == row["size"]:
            reuse_candidates.append({"live_path": live.relative_to(source_root).as_posix(),
                                     "backup_path": str(path), "size": row["size"],
                                     "verification": "Metadata candidate; full contents not checked"})
    spots = []
    for relative in matched:
        row = expected[relative]
        if len(spots) >= spot_check_count:
            break
        if row["size"] > spot_check_max_bytes or not re.fullmatch(r"[a-f0-9]{64}", row["hash"]):
            continue
        backup_digest = _bounded_hash(found[relative], spot_check_max_bytes)
        live = resolve_live_path(source_root, relative, moves)
        live_digest = None
        if live is not None and live.is_file() and live.stat().st_size <= spot_check_max_bytes:
            live_digest = _bounded_hash(live, spot_check_max_bytes)
        spots.append({"path": relative, "backup_sha256": backup_digest,
                      "backup_matches_inventory": backup_digest == row["hash"],
                      "live_path": str(live) if live is not None else None,
                      "live_matches_inventory": live_digest == row["hash"] if live_digest is not None else None})
    failed_spots = [entry["path"] for entry in spots if not entry["backup_matches_inventory"]]
    return {"state": "coverage_checked" if not (missing or unexpected or mismatches or failed_spots) else "incomplete_or_mismatched",
            "backup_root": str(backup_root), "inventory": str(Path(inventory).resolve()),
            "max_file_bytes": max_file_bytes, "expected_files": len(expected), "found_files": len(found),
            "metadata_matched_files": len(matched), "metadata_matched_bytes": sum(expected[path]["size"] for path in matched),
            "missing": missing, "unexpected": unexpected, "metadata_mismatches": mismatches,
            "auxiliary_metadata": [{"path": relative, "size": found[relative].lstat().st_size,
                                    "verification": "Presence/type only"} for relative in auxiliary],
            "spot_checks": spots, "spot_check_max_bytes": spot_check_max_bytes,
            "failed_spot_checks": failed_spots, "reuse_candidates": reuse_candidates,
            "reuse_candidate_count": len(reuse_candidates),
            "integrity_scope": "Coverage and bounded small-file SHA-256 only; remaining contents are unverified"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--inventory", type=Path, default=REPO_ROOT / "registry" / "ARTIFACT_INVENTORY.tsv")
    parser.add_argument("--moves", type=Path, default=REPO_ROOT / "registry" / "MOVES.tsv")
    parser.add_argument("--max-file-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--spot-check-count", type=int, default=10)
    parser.add_argument("--spot-check-max-bytes", type=int, default=1024**2)
    parser.add_argument("--summary-only", action="store_true", help="Omit the potentially long candidate path list")
    args = parser.parse_args(argv)
    try:
        report = coverage_report(args.backup_root, args.inventory, args.source_root, args.moves,
                                 args.max_file_bytes, args.spot_check_count, args.spot_check_max_bytes)
        if args.summary_only:
            report.pop("reuse_candidates")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["state"] == "coverage_checked" else 1
    except (OSError, ValueError, KeyError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
