"""Size-limited, content-verified snapshots; no source writes or network calls.

The policy uses JSON-compatible YAML so disaster recovery requires only Python's
standard library. Git history is an optional separate bundle, never a traversal
of .git. A completed snapshot is published atomically and made read-only.
"""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


class BackupError(RuntimeError):
    """The backup cannot be shown to be complete and unchanged."""


DEFAULT_POLICY = {
    "max_file_bytes": 2 * 1024**3,
    "required_mount": None,
    "free_space_margin_bytes": 0,
    "include_git_bundle": False,
    "exclude_names": [".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"],
    "exclude_patterns": [],
}
DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "configs" / "backup.yaml"
_STAT_KEYS = ("size", "mtime_ns", "ctime_ns", "mode", "device", "inode")


def load_policy(path):
    with Path(path).open(encoding="utf-8") as stream:
        supplied = json.load(stream)
    return _policy(supplied)


def _policy(supplied):
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_POLICY):
        raise BackupError("Unknown backup policy fields")
    policy = {**DEFAULT_POLICY, **supplied}
    for name in ("max_file_bytes", "free_space_margin_bytes"):
        value = policy[name]
        if type(value) is not int or value < (1 if name == "max_file_bytes" else 0):
            raise BackupError(f"{name} must be a valid positive byte count")
    for name in ("exclude_names", "exclude_patterns"):
        if not isinstance(policy[name], list) or not all(isinstance(item, str) for item in policy[name]):
            raise BackupError(f"{name} must contain strings")
    if ".git" not in policy["exclude_names"]:
        raise BackupError("Exclude .git and use the independent Git bundle")
    if type(policy["include_git_bundle"]) is not bool:
        raise BackupError("include_git_bundle must be true or false")
    if policy["required_mount"] is not None and not isinstance(policy["required_mount"], str):
        raise BackupError("required_mount must be a path or null")
    return policy


def _record(path, relative, info=None):
    info = path.lstat() if info is None else info
    kind = ("symlink" if stat.S_ISLNK(info.st_mode) else "directory" if stat.S_ISDIR(info.st_mode)
            else "file" if stat.S_ISREG(info.st_mode) else "special")
    record = {"path": relative, "kind": kind, "size": info.st_size,
              "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
              "mode": stat.S_IMODE(info.st_mode), "device": info.st_dev, "inode": info.st_ino}
    if kind == "symlink":
        record["target"] = os.readlink(path)
    return record


def _scan(root, policy=None):
    """Inventory lstat metadata, preserving directory/dangling links without following them."""
    included, excluded = [], []

    def visit(directory, prefix=""):
        with os.scandir(directory) as stream:
            children = sorted(stream, key=lambda item: item.name)
        for child in children:
            relative = f"{prefix}/{child.name}" if prefix else child.name
            path = Path(directory) / child.name
            record = _record(path, relative, child.stat(follow_symlinks=False))
            reason = None
            if policy is not None:
                if child.name in policy["exclude_names"]:
                    reason = "excluded_name"
                elif any(fnmatch.fnmatchcase(relative, pattern) for pattern in policy["exclude_patterns"]):
                    reason = "excluded_pattern"
                elif record["kind"] == "file" and record["size"] >= policy["max_file_bytes"]:
                    reason = "size_at_or_above_limit"
                elif record["kind"] == "special":
                    reason = "unsupported_file_type"
            if reason:
                excluded.append({**record, "reason": reason})
                continue
            included.append(record)
            if record["kind"] == "directory":
                visit(path, relative)

    visit(root)
    return included, excluded


def _summary(entries, excluded):
    return {"files": sum(entry["kind"] == "file" for entry in entries),
            "symlinks": sum(entry["kind"] == "symlink" for entry in entries),
            "directories": sum(entry["kind"] == "directory" for entry in entries),
            "bytes": sum(entry["size"] for entry in entries if entry["kind"] == "file"),
            "excluded_paths": len(excluded),
            "excluded_file_bytes": sum(entry["size"] for entry in excluded if entry["kind"] == "file")}


def plan_snapshot(root, policy=None, previous_snapshot=None):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise BackupError("Source root must be a directory")
    policy = _policy(policy or {})
    entries, excluded = _scan(root, policy)
    if not any(entry["kind"] in ("file", "symlink") for entry in entries):
        raise BackupError("Refusing an empty backup inventory")
    plan = {"schema_version": 1, "state": "planned", "source_root": str(root),
            "created_utc": datetime.now(timezone.utc).isoformat(), "policy": policy,
            "root_metadata": _record(root, "."), "entries": entries, "excluded": excluded,
            "summary": _summary(entries, excluded)}
    if previous_snapshot is not None:
        previous_snapshot = Path(previous_snapshot).resolve(strict=True)
        previous = _load_manifest(previous_snapshot)
        previous_files = {entry["path"]: entry for entry in previous["entries"] if entry["kind"] == "file"}
        candidates = [entry["path"] for entry in entries if entry["kind"] == "file"
                      and entry["path"] in previous_files and entry["size"] == previous_files[entry["path"]]["size"]]
        plan["reuse"] = {"snapshot": str(previous_snapshot),
                        "manifest_sha256": _hash_file(previous_snapshot / "manifest.json"),
                        "candidate_paths": candidates,
                        "status": "Candidates only; source SHA-256 and referenced bytes must verify before completion"}
    return plan


def _open_regular(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise BackupError(f"Expected a regular file: {path}")
    return os.fdopen(fd, "rb")


def _hash_file(path):
    digest = hashlib.sha256()
    with _open_regular(path) as stream:
        while chunk := stream.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _check_source(path, expected, info=None):
    actual = _record(path, expected["path"], info)
    if actual["kind"] != expected["kind"] or any(actual[key] != expected[key] for key in _STAT_KEYS):
        raise BackupError(f"Source changed during backup: {path}")
    if expected["kind"] == "symlink" and actual["target"] != expected["target"]:
        raise BackupError(f"Symlink changed during backup: {path}")


def _copy_regular(source, destination, expected=None):
    digest = hashlib.sha256()
    with _open_regular(source) as stream:
        if expected is not None:
            _check_source(source, expected, os.fstat(stream.fileno()))
        with destination.open("xb") as output:
            while chunk := stream.read(4 * 1024**2):
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if expected is not None:
            _check_source(source, expected, os.fstat(stream.fileno()))
            _check_source(source, expected)
    return digest.hexdigest()


def _json_write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_paths(entries):
    if not isinstance(entries, list):
        raise BackupError("Invalid snapshot inventory")
    records = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise BackupError("Invalid snapshot inventory entry")
        path = entry.get("path", "")
        if not isinstance(path, str):
            raise BackupError("Manifest paths must be strings")
        parsed = PurePosixPath(path)
        if (not path or parsed.is_absolute() or ".." in parsed.parts
                or path != parsed.as_posix() or path == "." or "\x00" in path):
            raise BackupError(f"Unsafe manifest path: {path!r}")
        if path in records or entry.get("kind") not in ("file", "directory", "symlink"):
            raise BackupError(f"Duplicate or invalid inventory entry: {path}")
        if type(entry.get("mode")) is not int or not 0 <= entry["mode"] <= 0o7777:
            raise BackupError(f"Invalid permission metadata: {path}")
        if entry["kind"] == "file":
            if type(entry.get("size")) is not int or entry["size"] < 0:
                raise BackupError(f"Invalid file size: {path}")
            if not re.fullmatch(r"[a-f0-9]{64}", entry.get("sha256", "")):
                raise BackupError(f"Missing complete SHA-256: {path}")
        if entry["kind"] == "symlink" and not isinstance(entry.get("target"), str):
            raise BackupError(f"Missing symlink target: {path}")
        if "reference" in entry:
            reference = entry["reference"]
            if (entry["kind"] != "file" or not isinstance(reference, dict)
                    or set(reference) != {"snapshot", "path", "manifest_sha256"}):
                raise BackupError(f"Invalid snapshot reference: {path}")
            if (not isinstance(reference["snapshot"], str) or not Path(reference["snapshot"]).is_absolute()
                    or not isinstance(reference["path"], str)
                    or not re.fullmatch(r"[a-f0-9]{64}", reference.get("manifest_sha256", ""))):
                raise BackupError(f"Invalid snapshot reference metadata: {path}")
            reference_path = PurePosixPath(reference["path"])
            if (reference_path.is_absolute() or ".." in reference_path.parts
                    or reference["path"] != reference_path.as_posix() or reference["path"] == "."):
                raise BackupError(f"Unsafe snapshot reference path: {path}")
        records[path] = entry
    if not any(entry["kind"] in ("file", "symlink") for entry in entries):
        raise BackupError("Refusing an empty snapshot manifest")
    for path in records:
        for parent in PurePosixPath(path).parents:
            if str(parent) != "." and records.get(str(parent), {}).get("kind") != "directory":
                raise BackupError(f"Manifest parent is absent or is a symlink: {path}")
    return records


def _load_manifest(snapshot):
    snapshot = Path(snapshot)
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise BackupError("Snapshot must be a real directory")
    with _open_regular(snapshot / "manifest.sha256") as stream:
        expected_digest = stream.read().decode("ascii").split()
    if len(expected_digest) != 2 or expected_digest[1] != "manifest.json":
        raise BackupError("Invalid manifest checksum record")
    if _hash_file(snapshot / "manifest.json") != expected_digest[0]:
        raise BackupError("Snapshot manifest checksum mismatch")
    with _open_regular(snapshot / "manifest.json") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1 or manifest.get("state") != "complete":
        raise BackupError("Snapshot is incomplete or has an unsupported schema")
    _validate_paths(manifest.get("entries"))
    return manifest


def _resolve_file(snapshot, entry, seen=None):
    seen = set() if seen is None else seen
    key = (str(Path(snapshot).resolve()), entry["path"])
    if key in seen or len(seen) >= 32:
        raise BackupError("Cyclic or excessively deep snapshot reference")
    seen.add(key)
    reference = entry.get("reference")
    if reference is None:
        payload = Path(snapshot) / "files"
        # A changed parent link must not redirect reads outside a sealed payload.
        path = payload
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise BackupError("Referenced payload is missing or is a symlink")
        for component in PurePosixPath(entry["path"]).parts[:-1]:
            path = path / component
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise BackupError("Referenced file parent is absent or is a symlink")
        return payload / entry["path"]
    previous = Path(reference["snapshot"])
    manifest = _load_manifest(previous)
    if _hash_file(previous / "manifest.json") != reference["manifest_sha256"]:
        raise BackupError("Referenced snapshot manifest changed")
    candidates = {item["path"]: item for item in manifest["entries"]}
    previous_entry = candidates.get(reference["path"])
    if (previous_entry is None or previous_entry["kind"] != "file"
            or (previous_entry["size"], previous_entry["sha256"]) != (entry["size"], entry["sha256"])):
        raise BackupError("Referenced file does not match snapshot inventory")
    return _resolve_file(previous, previous_entry, seen)


def _verify_payload(payload, entries, snapshot=None):
    if not payload.is_dir() or payload.is_symlink():
        raise BackupError("Snapshot payload is missing or is a symlink")
    actual, _ = _scan(payload)
    actual = {entry["path"]: entry for entry in actual}
    expected = {entry["path"]: entry for entry in entries if snapshot is None or "reference" not in entry}
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise BackupError(f"Snapshot coverage mismatch: missing={missing}, unexpected={extra}")
    checked_bytes = 0
    for entry in entries:
        relative = entry["path"]
        if "reference" not in entry or snapshot is None:
            found = actual[relative]
            if found["kind"] != entry["kind"]:
                raise BackupError(f"Snapshot type mismatch: {relative}")
            if entry["kind"] == "symlink" and found["target"] != entry["target"]:
                raise BackupError(f"Snapshot symlink target mismatch: {relative}")
        if entry["kind"] == "file":
            file_path = _resolve_file(snapshot, entry) if snapshot is not None else payload / relative
            info = file_path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size != entry["size"]:
                raise BackupError(f"Snapshot file type/size mismatch: {relative}")
            if _hash_file(file_path) != entry["sha256"]:
                raise BackupError(f"Snapshot file SHA-256 mismatch: {relative}")
            checked_bytes += info.st_size
    return checked_bytes


def verify_snapshot(snapshot):
    """Check completeness and full contents without consulting the live source tree."""
    snapshot = Path(snapshot).resolve(strict=True)
    manifest = _load_manifest(snapshot)
    expected_top = {"files", "manifest.json", "manifest.sha256"}
    bundle = manifest.get("git_bundle")
    if bundle is not None:
        expected_top.add("repository.bundle")
        bundle_path = snapshot / "repository.bundle"
        if bundle_path.lstat().st_size != bundle["size"] or _hash_file(bundle_path) != bundle["sha256"]:
            raise BackupError("Git bundle integrity mismatch")
    if {path.name for path in snapshot.iterdir()} != expected_top:
        raise BackupError("Unexpected or missing snapshot metadata files")
    checked_bytes = _verify_payload(snapshot / "files", manifest["entries"], snapshot)
    return {"state": "verified", "snapshot": str(snapshot),
            "entries": len(manifest["entries"]), "verified_file_bytes": checked_bytes,
            "referenced_files": sum("reference" in entry for entry in manifest["entries"]),
            "excluded_paths": len(manifest["excluded"])}


def _git(root, *arguments):
    result = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *arguments],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _check_destination(root, destination, policy, required_bytes):
    destination = destination.resolve()
    if destination == root or root in destination.parents:
        raise BackupError("Backup destination must be outside the source tree")
    mount = policy["required_mount"]
    if mount is not None:
        mount = Path(mount).resolve()
        if not os.path.ismount(mount) or not (destination == mount or mount in destination.parents):
            raise BackupError(f"Required backup disk is not mounted at {mount}")
    existing = destination
    while not existing.exists():
        existing = existing.parent
    if not existing.is_dir():
        raise BackupError("Backup destination parent must be a directory")
    if shutil.disk_usage(existing).free < required_bytes + policy["free_space_margin_bytes"]:
        raise BackupError("Insufficient free space for snapshot and configured margin")
    return destination


def _seal(snapshot):
    directories = []
    for root, dirs, files in os.walk(snapshot, followlinks=False):
        directories.append(Path(root))
        for name in files:
            path = Path(root) / name
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    for path in reversed(directories):
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_snapshot(root, destination, policy=None, source_plan=None, label="snapshot",
                    git_bundle=None, previous_snapshot=None, progress=None):
    """Create a new snapshot; aborted .partial directories are never published as complete."""
    root = Path(root).resolve(strict=True)
    policy = _policy(policy or {})
    plan = plan_snapshot(root, policy, previous_snapshot)
    if source_plan is not None:
        if (source_plan.get("source_root") != str(root) or source_plan.get("policy") != policy
                or source_plan.get("entries") != plan["entries"]):
            raise BackupError("Source no longer matches the approved preflight plan")
    destination = _check_destination(root, Path(destination), policy, plan["summary"]["bytes"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", label):
        raise BackupError("Snapshot label must be a short filename-safe string")
    previous_files = {}
    previous_digest = None
    if previous_snapshot is not None:
        previous_snapshot = Path(previous_snapshot).resolve(strict=True)
        previous = _load_manifest(previous_snapshot)
        previous_files = {entry["path"]: entry for entry in previous["entries"] if entry["kind"] == "file"}
        previous_digest = _hash_file(previous_snapshot / "manifest.json")
    git_bundle = policy["include_git_bundle"] if git_bundle is None else git_bundle
    git_state = {"head": _git(root, "rev-parse", "HEAD"),
                 "status_porcelain": _git(root, "status", "--porcelain=v1", "--untracked-files=all")} if git_bundle else None
    name = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{label}-{uuid.uuid4().hex[:8]}"
    destination.mkdir(parents=True, exist_ok=True)
    stage = destination / f".{name}.partial"
    stage.mkdir()
    payload = stage / "files"
    payload.mkdir()
    manifest = {**plan, "state": "complete", "git_state": git_state, "git_bundle": None}
    try:
        if git_bundle:
            bundle_path = stage / "repository.bundle"
            _git(root, "bundle", "create", str(bundle_path), "--all", "HEAD")
            _git(root, "bundle", "verify", str(bundle_path))
            manifest["git_bundle"] = {"size": bundle_path.stat().st_size, "sha256": _hash_file(bundle_path)}
        for entry in plan["entries"]:
            if entry["kind"] == "directory":
                (payload / entry["path"]).mkdir()
        records = []
        leaves = [entry for entry in plan["entries"] if entry["kind"] != "directory"]
        # Small source/configuration files are protected before bulky empirical artifacts.
        leaves.sort(key=lambda entry: (entry["size"], entry["path"]))
        for index, entry in enumerate(leaves, 1):
            source, target = root / entry["path"], payload / entry["path"]
            _check_source(source, entry)
            record = dict(entry)
            if entry["kind"] == "symlink":
                target.symlink_to(entry["target"])
                _check_source(source, entry)
            else:
                previous = previous_files.get(entry["path"])
                if previous is not None and previous["size"] == entry["size"]:
                    digest = _hash_file(source)
                    _check_source(source, entry)
                    if digest == previous["sha256"]:
                        record["sha256"] = digest
                        record["reference"] = {"snapshot": str(previous_snapshot), "path": entry["path"],
                                               "manifest_sha256": previous_digest}
                if "reference" not in record:
                    record["sha256"] = _copy_regular(source, target, entry)
                    target.chmod(entry["mode"])
                    os.utime(target, ns=(entry["mtime_ns"], entry["mtime_ns"]))
            records.append(record)
            if progress is not None:
                progress(index, len(leaves), entry["path"])
        records.extend(entry for entry in plan["entries"] if entry["kind"] == "directory")
        manifest["entries"] = sorted(records, key=lambda entry: entry["path"])
        after, _ = _scan(root, policy)
        if after != plan["entries"]:
            raise BackupError("Source inventory changed during backup; snapshot not published")
        if git_bundle and ({"head": _git(root, "rev-parse", "HEAD"),
                            "status_porcelain": _git(root, "status", "--porcelain=v1", "--untracked-files=all")} != git_state):
            raise BackupError("Git state changed during backup")
        _json_write(stage / "manifest.json", manifest)
        with (stage / "manifest.sha256").open("x", encoding="ascii") as stream:
            stream.write(f"{_hash_file(stage / 'manifest.json')}  manifest.json\n")
            stream.flush()
            os.fsync(stream.fileno())
        verify_snapshot(stage)
        _seal(stage)
        for directory, _, _ in os.walk(stage, followlinks=False):
            _fsync_directory(directory)
        final = destination / name
        if os.path.lexists(final):
            raise BackupError("Snapshot destination already exists")
        stage.rename(final)
        _fsync_directory(destination)
        return final
    except Exception as error:
        raise BackupError(f"Snapshot incomplete; unpublished files retained at {stage}: {error}") from error


def restore_snapshot(snapshot, destination):
    """Restore to a fresh directory, never into an existing working tree."""
    snapshot = Path(snapshot).resolve(strict=True)
    destination = Path(destination).absolute()
    if os.path.lexists(destination):
        raise BackupError("Restore destination already exists; choose a fresh directory")
    if snapshot == destination.resolve() or snapshot in destination.resolve().parents:
        raise BackupError("Restore destination cannot be inside its snapshot")
    if not destination.parent.is_dir():
        raise BackupError("Restore destination parent must already exist")
    verify_snapshot(snapshot)
    manifest = _load_manifest(snapshot)
    stage = destination.parent / f".{destination.name}.restoring-{uuid.uuid4().hex[:8]}"
    stage.mkdir()
    try:
        directories = sorted((entry for entry in manifest["entries"] if entry["kind"] == "directory"),
                             key=lambda entry: len(PurePosixPath(entry["path"]).parts))
        for entry in directories:
            (stage / entry["path"]).mkdir()
        for entry in manifest["entries"]:
            target = stage / entry["path"]
            if entry["kind"] == "symlink":
                target.symlink_to(entry["target"])
            elif entry["kind"] == "file":
                digest = _copy_regular(_resolve_file(snapshot, entry), target)
                if digest != entry["sha256"]:
                    raise BackupError(f"Restored SHA-256 mismatch: {entry['path']}")
                target.chmod(entry["mode"])
                os.utime(target, ns=(entry["mtime_ns"], entry["mtime_ns"]))
        _verify_payload(stage, manifest["entries"])
        for entry in reversed(directories):
            target = stage / entry["path"]
            target.chmod(entry["mode"])
            os.utime(target, ns=(entry["mtime_ns"], entry["mtime_ns"]))
        stage.chmod(manifest["root_metadata"]["mode"])
        if os.path.lexists(destination):
            raise BackupError("Restore destination appeared during verification")
        stage.rename(destination)
        _fsync_directory(destination.parent)
        return destination
    except Exception as error:
        raise BackupError(f"Restore incomplete; unpublished files retained at {stage}: {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "create"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True, type=Path)
        command.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
        command.add_argument("--previous-snapshot", type=Path)
        if name == "plan":
            command.add_argument("--output", type=Path)
        else:
            command.add_argument("--destination", type=Path, required=True)
            command.add_argument("--source-plan", type=Path)
            command.add_argument("--label", default="snapshot")
            command.add_argument("--git-bundle", action=argparse.BooleanOptionalAction, default=None)
    for name in ("verify", "restore"):
        command = commands.add_parser(name)
        command.add_argument("--snapshot", type=Path, required=True)
        if name == "restore":
            command.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan_snapshot(args.root, load_policy(args.policy), args.previous_snapshot)
            if args.output:
                if args.root.resolve() == args.output.resolve() or args.root.resolve() in args.output.resolve().parents:
                    raise BackupError("Save preflight plans outside the source tree")
                _json_write(args.output, result)
        elif args.command == "create":
            plan = json.loads(args.source_plan.read_text()) if args.source_plan else None

            def progress(index, total, path):
                if index % 100 == 0 or index == total:
                    print(f"Copied or referenced {index}/{total}: {path}", file=sys.stderr, flush=True)

            created = create_snapshot(args.root, args.destination, load_policy(args.policy), plan,
                                      args.label, args.git_bundle, args.previous_snapshot, progress)
            result = {"state": "complete", "snapshot": str(created)}
        elif args.command == "verify":
            result = verify_snapshot(args.snapshot)
        else:
            result = {"state": "restored", "destination": str(restore_snapshot(args.snapshot, args.destination))}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (BackupError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
