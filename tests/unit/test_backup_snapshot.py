"""Backup guarantees using tiny temporary files; no empirical artifacts or network."""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops.backup import snapshot

pytestmark = pytest.mark.cpu


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (root / ".gitignore").write_text("*.cache\n")
    (root / "subdir").mkdir()
    (root / "subdir" / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "subdir" / "run.sh").chmod(0o755)
    (root / "subdir" / "empty").mkdir()
    (root / "odd\nname.tsv").write_text("a\tb\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "bytecode.pyc").write_bytes(b"ignored")
    (root / ".git").mkdir()
    (root / ".git" / "ignored").write_bytes(b"ignored")
    return root


def _unseal(root):
    for directory, _, files in os.walk(root, followlinks=False):
        Path(directory).chmod(0o755)
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(0o644)


def _rewrite_manifest(path, manifest):
    _unseal(path)
    (path / "manifest.json").write_text(json.dumps(manifest))
    digest = hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
    (path / "manifest.sha256").write_text(f"{digest}  manifest.json\n")


def test_plan_covers_hidden_config_and_records_exact_limit_exclusion(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / ".gitignore").write_bytes(b"1234567")
    (root / "large.bin").write_bytes(b"12345678")
    plan = snapshot.plan_snapshot(root, {"max_file_bytes": 8})
    assert [entry["path"] for entry in plan["entries"]] == [".gitignore"]
    assert plan["excluded"][0]["reason"] == "size_at_or_above_limit"
    assert plan["excluded"][0]["size"] == 8
    assert plan["summary"]["bytes"] == 7


def test_snapshot_restore_preserves_bytes_links_and_modes_without_following_targets(source, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "do_not_copy.bin").write_bytes(b"external data")
    (source / "external").symlink_to(external, target_is_directory=True)
    (source / "dangling").symlink_to("missing")
    original_mode = stat.S_IMODE((source / "subdir" / "run.sh").stat().st_mode)
    created = snapshot.create_snapshot(source, tmp_path / "backups")
    assert snapshot.verify_snapshot(created)["state"] == "verified"
    assert not (created / "files" / "__pycache__").exists()
    assert not (created / "files" / ".git").exists()
    assert (created / "files" / "external").is_symlink()
    manifest = json.loads((created / "manifest.json").read_text())
    assert not any(entry["path"].startswith("external/") for entry in manifest["entries"])
    assert not stat.S_IMODE((created / "files" / "pyproject.toml").stat().st_mode) & 0o222
    assert stat.S_IMODE((source / "subdir" / "run.sh").stat().st_mode) == original_mode
    restored = snapshot.restore_snapshot(created, tmp_path / "restored")
    assert (restored / "pyproject.toml").read_bytes() == (source / "pyproject.toml").read_bytes()
    assert (restored / "odd\nname.tsv").read_text() == "a\tb\n"
    assert (restored / "subdir" / "empty").is_dir()
    assert stat.S_IMODE((restored / "subdir" / "run.sh").stat().st_mode) == 0o755
    assert os.readlink(restored / "external") == str(external)
    assert os.readlink(restored / "dangling") == "missing"


@pytest.mark.parametrize("failure", ["deleted", "unexpected", "same_size_corruption", "link_changed"])
def test_independent_verification_rejects_coverage_and_content_changes(source, tmp_path, failure):
    (source / "link").symlink_to("pyproject.toml")
    created = snapshot.create_snapshot(source, tmp_path / "backups")
    _unseal(created)
    path = created / "files" / "pyproject.toml"
    if failure == "deleted":
        path.unlink()
    elif failure == "unexpected":
        (created / "files" / "extra.txt").write_text("extra")
    elif failure == "same_size_corruption":
        path.write_bytes(b"x" * path.stat().st_size)
    else:
        (created / "files" / "link").unlink()
        (created / "files" / "link").symlink_to("wrong")
    with pytest.raises(snapshot.BackupError):
        snapshot.verify_snapshot(created)
    with pytest.raises(snapshot.BackupError):
        snapshot.restore_snapshot(created, tmp_path / "should_not_exist")
    assert not (tmp_path / "should_not_exist").exists()


def test_manifest_corruption_and_empty_snapshot_are_rejected(source, tmp_path):
    created = snapshot.create_snapshot(source, tmp_path / "backups")
    _unseal(created)
    (created / "manifest.json").write_text("{}")
    with pytest.raises(snapshot.BackupError, match="checksum mismatch"):
        snapshot.verify_snapshot(created)
    manifest = {"schema_version": 1, "state": "complete", "entries": []}
    _rewrite_manifest(created, manifest)
    with pytest.raises(snapshot.BackupError, match="empty snapshot"):
        snapshot.verify_snapshot(created)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(snapshot.BackupError, match="empty backup"):
        snapshot.plan_snapshot(empty)


@pytest.mark.parametrize("path", ["../outside", "/outside", "subdir/../../outside", "subdir//file"])
def test_restore_rejects_path_escape_even_with_matching_manifest_digest(source, tmp_path, path):
    created = snapshot.create_snapshot(source, tmp_path / "backups")
    manifest = json.loads((created / "manifest.json").read_text())
    manifest["entries"][0]["path"] = path
    _rewrite_manifest(created, manifest)
    with pytest.raises(snapshot.BackupError, match="Unsafe manifest path"):
        snapshot.restore_snapshot(created, tmp_path / "restored")
    assert not (tmp_path / "outside").exists()


def test_restore_never_overwrites_existing_directory_or_dangling_link(source, tmp_path):
    created = snapshot.create_snapshot(source, tmp_path / "backups")
    destination = tmp_path / "work"
    destination.mkdir()
    protected = destination / "keep.txt"
    protected.write_text("new work")
    with pytest.raises(snapshot.BackupError, match="already exists"):
        snapshot.restore_snapshot(created, destination)
    assert protected.read_text() == "new work"
    link = tmp_path / "dangling_restore"
    link.symlink_to("missing")
    with pytest.raises(snapshot.BackupError, match="already exists"):
        snapshot.restore_snapshot(created, link)


def test_mount_and_space_guards_happen_before_destination_writes(source, tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot.os.path, "ismount", lambda path: False)
    destination = tmp_path / "backups"
    with pytest.raises(snapshot.BackupError, match="not mounted"):
        snapshot.create_snapshot(source, destination, {"required_mount": str(tmp_path)})
    assert not destination.exists()
    monkeypatch.setattr(snapshot.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(snapshot.BackupError, match="Insufficient free space"):
        snapshot.create_snapshot(source, destination)
    assert not destination.exists()
    with pytest.raises(snapshot.BackupError, match="outside the source"):
        snapshot.create_snapshot(source, source / "backups")


def test_source_mutation_never_publishes_completed_snapshot(source, tmp_path, monkeypatch):
    original_copy = snapshot._copy_regular
    changed = False

    def mutate(source_path, destination_path, expected=None):
        nonlocal changed
        result = original_copy(source_path, destination_path, expected)
        if not changed:
            source_path.write_bytes(source_path.read_bytes() + b"changed")
            changed = True
        return result

    monkeypatch.setattr(snapshot, "_copy_regular", mutate)
    destination = tmp_path / "backups"
    with pytest.raises(snapshot.BackupError, match="Source inventory changed"):
        snapshot.create_snapshot(source, destination)
    assert all(path.name.endswith(".partial") for path in destination.iterdir())


def test_approved_source_plan_rejects_changed_inputs(source, tmp_path):
    plan = snapshot.plan_snapshot(source)
    (source / "pyproject.toml").write_text("changed")
    with pytest.raises(snapshot.BackupError, match="approved preflight"):
        snapshot.create_snapshot(source, tmp_path / "backups", source_plan=plan)
    assert not (tmp_path / "backups").exists()


def test_incremental_references_are_hashed_pinned_and_restorable(source, tmp_path):
    first = snapshot.create_snapshot(source, tmp_path / "backups")
    original = (first / "files" / "pyproject.toml").read_bytes()
    (source / ".gitignore").write_text("changed rules\n")
    plan = snapshot.plan_snapshot(source, previous_snapshot=first)
    assert "pyproject.toml" in plan["reuse"]["candidate_paths"]
    second = snapshot.create_snapshot(source, tmp_path / "backups", previous_snapshot=first)
    assert snapshot.verify_snapshot(second)["referenced_files"] == 3
    assert not (second / "files" / "pyproject.toml").exists()
    restored = snapshot.restore_snapshot(second, tmp_path / "restored")
    assert (restored / "pyproject.toml").read_bytes() == original
    assert (restored / ".gitignore").read_text() == "changed rules\n"
    # A referenced ancestor remains a required dependency and cannot silently drift.
    _unseal(first)
    (first / "files" / "pyproject.toml").write_bytes(b"x" * len(original))
    with pytest.raises(snapshot.BackupError, match="SHA-256 mismatch"):
        snapshot.verify_snapshot(second)


def test_incremental_reference_cannot_follow_replaced_parent_symlink(source, tmp_path):
    first = snapshot.create_snapshot(source, tmp_path / "backups")
    second = snapshot.create_snapshot(source, tmp_path / "backups", previous_snapshot=first)
    _unseal(first)
    original_subdir = first / "files" / "subdir"
    outside = tmp_path / "redirected"
    original_subdir.rename(outside)
    original_subdir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(snapshot.BackupError, match="parent is absent or is a symlink"):
        snapshot.verify_snapshot(second)


def test_incremental_reference_path_escape_is_rejected(source, tmp_path):
    first = snapshot.create_snapshot(source, tmp_path / "backups")
    second = snapshot.create_snapshot(source, tmp_path / "backups", previous_snapshot=first)
    manifest = json.loads((second / "manifest.json").read_text())
    entry = next(entry for entry in manifest["entries"] if "reference" in entry)
    entry["reference"]["path"] = "../outside"
    _rewrite_manifest(second, manifest)
    with pytest.raises(snapshot.BackupError, match="Unsafe snapshot reference path"):
        snapshot.verify_snapshot(second)


@pytest.mark.skipif(shutil.which("git") is None, reason="Git executable unavailable")
def test_git_bundle_recovers_history_and_snapshot_retains_uncommitted_files(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()

    def git(*arguments):
        return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *arguments],
                              check=True, capture_output=True, text=True).stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Backup fixture")
    git("config", "user.email", "fixture@example.invalid")
    (root / "tracked.py").write_text("baseline = True\n")
    git("add", "tracked.py")
    git("-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture baseline")
    commit = git("rev-parse", "HEAD")
    (root / "tracked.py").write_text("uncommitted = True\n")
    (root / "untracked.txt").write_text("present in source snapshot\n")
    created = snapshot.create_snapshot(root, tmp_path / "backups", git_bundle=True)
    assert (created / "files" / "tracked.py").read_text() == "uncommitted = True\n"
    assert (created / "files" / "untracked.txt").exists()
    clone = tmp_path / "recovered_git"
    subprocess.run(["git", "clone", "--quiet", str(created / "repository.bundle"), str(clone)],
                   check=True, capture_output=True, text=True)
    recovered = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"],
                               check=True, capture_output=True, text=True).stdout.strip()
    assert recovered == commit
    assert (clone / "tracked.py").read_text() == "baseline = True\n"


def test_module_and_standalone_cli_are_available_without_import_path_hacks(source, tmp_path):
    policy = tmp_path / "policy.yaml"
    policy.write_text(json.dumps(snapshot.DEFAULT_POLICY))
    repo = Path(__file__).resolve().parents[2]
    for invocation in ([sys.executable, "-m", "ops.backup.snapshot"],
                       [sys.executable, str(repo / "ops" / "backup" / "snapshot.py")]):
        result = subprocess.run([*invocation, "plan", "--root", str(source), "--policy", str(policy)],
                                cwd=repo, check=True, capture_output=True, text=True)
        assert json.loads(result.stdout)["state"] == "planned"
