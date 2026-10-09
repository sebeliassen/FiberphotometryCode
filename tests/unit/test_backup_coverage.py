"""Existing mirror coverage uses only tiny fixtures and bounded hashes."""

import csv
import hashlib

import pytest

from ops.backup import coverage, verify_tier_a_backup

pytestmark = pytest.mark.cpu


@pytest.fixture
def mirror(tmp_path):
    live, backup = tmp_path / "live", tmp_path / "backup"
    (live / "new").mkdir(parents=True)
    (backup / "old").mkdir(parents=True)
    contents = b"fixture\n"
    (live / "new" / "small.txt").write_bytes(contents)
    (backup / "old" / "small.txt").write_bytes(contents)
    inventory = tmp_path / "inventory.tsv"
    with inventory.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["path", "size", "tier", "hash"])
        writer.writerow(["old/small.txt", len(contents), "A", hashlib.sha256(contents).hexdigest()])
        writer.writerow(["old/large.pkl", 4096, "A", "ht16m:partial"])
    moves = tmp_path / "moves.tsv"
    moves.write_text("source_path\ttarget_path\nold\tintermediate\nintermediate\tnew\n")
    return live, backup, inventory, moves


def test_coverage_reports_moved_paths_and_bounded_full_hash_scope(mirror):
    live, backup, inventory, moves = mirror
    report = coverage.coverage_report(backup, inventory, live, moves, max_file_bytes=128,
                                      spot_check_count=2, spot_check_max_bytes=16)
    assert report["state"] == "coverage_checked"
    assert report["expected_files"] == 1
    assert report["metadata_matched_bytes"] == 8
    assert report["spot_checks"][0]["backup_matches_inventory"] is True
    assert report["spot_checks"][0]["live_matches_inventory"] is True
    assert report["reuse_candidates"][0]["live_path"] == "new/small.txt"
    assert "remaining contents are unverified" in report["integrity_scope"]


def test_missing_and_unexpected_mirror_files_do_not_pass(mirror):
    live, backup, inventory, moves = mirror
    (backup / "old" / "small.txt").unlink()
    (backup / "unexpected.txt").write_text("extra")
    report = coverage.coverage_report(backup, inventory, live, moves, max_file_bytes=128)
    assert report["state"] == "incomplete_or_mismatched"
    assert report["missing"] == ["old/small.txt"]
    assert report["unexpected"] == ["unexpected.txt"]


def test_known_phase_zero_provenance_is_reported_separately_from_payload(mirror):
    live, backup, inventory, moves = mirror
    (backup / "BACKUP_REPORT.md").write_text("phase zero metadata")
    report = coverage.coverage_report(backup, inventory, live, moves, max_file_bytes=128)
    assert report["state"] == "coverage_checked"
    assert report["metadata_matched_files"] == 1
    assert report["found_files"] == 2
    assert report["unexpected"] == []
    assert report["auxiliary_metadata"][0]["path"] == "BACKUP_REPORT.md"


def test_spot_check_reports_backup_corruption_and_live_drift_separately(mirror):
    live, backup, inventory, moves = mirror
    (live / "new" / "small.txt").write_bytes(b"changed\n")
    report = coverage.coverage_report(backup, inventory, live, moves, max_file_bytes=128)
    assert report["state"] == "coverage_checked"
    assert report["spot_checks"][0]["live_matches_inventory"] is False
    (backup / "old" / "small.txt").write_bytes(b"changed\n")
    report = coverage.coverage_report(backup, inventory, live, moves, max_file_bytes=128)
    assert report["state"] == "incomplete_or_mismatched"
    assert report["failed_spot_checks"] == ["old/small.txt"]


def test_spot_checks_never_read_files_above_bound(mirror, monkeypatch):
    live, backup, inventory, moves = mirror
    monkeypatch.setattr(coverage, "_bounded_hash", lambda *args: pytest.fail("Must not read content"))
    report = coverage.coverage_report(backup, inventory, live, moves, max_file_bytes=128,
                                      spot_check_max_bytes=4)
    assert report["spot_checks"] == []


def test_empty_backup_or_empty_expected_inventory_is_rejected(mirror):
    live, backup, inventory, moves = mirror
    with pytest.raises(ValueError, match="inventory is empty"):
        coverage.coverage_report(backup, inventory, live, moves, max_file_bytes=8)
    (backup / "old" / "small.txt").unlink()
    with pytest.raises(ValueError, match="backup is empty"):
        coverage.coverage_report(backup, inventory, live, moves, max_file_bytes=128)


def test_moves_cannot_redirect_live_spot_checks_outside_source(mirror, tmp_path):
    live, _, _, _ = mirror
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data.txt").write_text("external data")
    (live / "escape").symlink_to(outside, target_is_directory=True)
    assert coverage.resolve_live_path(live, "escape/data.txt", []) is None


def test_legacy_verifier_refuses_incomplete_expected_inventory_before_hashing(mirror, monkeypatch):
    live, backup, inventory, moves = mirror
    monkeypatch.setattr(verify_tier_a_backup, "BACKUP_ROOT", backup)
    monkeypatch.setattr(verify_tier_a_backup, "REPO_ROOT", live)
    monkeypatch.setattr(verify_tier_a_backup, "INVENTORY_PATH", inventory)
    monkeypatch.setattr(verify_tier_a_backup, "MOVES_PATH", moves)
    monkeypatch.setattr(verify_tier_a_backup, "compute_content_sha256",
                        lambda path: pytest.fail("Must not hash before checking missing files"))
    with pytest.raises(SystemExit) as error:
        verify_tier_a_backup.main()
    assert error.value.code == 1
