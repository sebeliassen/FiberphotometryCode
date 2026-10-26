# Recovery and verification

Current maintained source is on `main`. Original modular (`31326cc`) and Karen
seqNMF delivery (`2c43b75`) source are part of its history. Older branch names
may have been retired; use preserved tags or Git commits to inspect those states.
`codex/pre-consolidation-20261026` preserves the old primary tip `16c5337`,
including its unique task notes, and is available on origin.

Source/history bundles and restore evidence from the integration are retained
under the primary checkout's `outputs/consolidation_20261021`. For example,
`preserved_all_refs.bundle` records the original five local branches. Use the
[current task state](tasks/CURRENT.md) for the latest source revision/checks.
Inspect older source with `git show COMMIT:path`; restore into a separate empty
location when needed, rather than resetting a live runtime checkout.

Git snapshots do not restore ignored recordings, model/session payloads,
generated review products or browser-local feedback. Symlinks preserve a target
name, not a backup of that target. Keep current empirical storage and both review
viewers' outputs intact; source cleanup does not authorize their removal.

## Existing HDD backups

- `/mnt/hdd/backups/code_snapshots/20261009_112705/code_snapshot.tar.gz`:
  existing archive's SHA-256 verified against its checksum file.
- `/mnt/hdd/backups/TierA_backup_20261009/`: 509 expected eligible Tier-A
  files, 47,480,759,461 bytes. Path/type/size coverage passed and ten files
  of at most 1 MiB matched recorded full SHA-256 hashes and relocated live files.
  This is not full verification of all 44.22 GiB.
- `/mnt/hdd/backups/FiberphotometryCode/20261009_153903-baseline/`:
  separately verified current-Gemini source archive and Git bundle, recorded in
  `code_baseline.json`. Its `workspace/` copy was cancelled as redundant and is
  incomplete. `manifest.json` explicitly records `CANCELLED_INCOMPLETE`.

The October 9 final-refactor backup and restore drill were initially deferred.
A later sealed snapshot now exists at
`/mnt/hdd/backups/FiberphotometryCode/20261011T143033Z-snapshot-353ab324/`.
Its manifest records Git `e7010dd`, 96,373,603,551 bytes across 8,440 files,
1,316 directories and 330 symlinks (10,086 entries), plus a separately hashed
702,166,666-byte Git history bundle. Creation verifies the included payload
before sealing and publication. A metadata-only review confirmed the manifest
checksum and read-only snapshot root; it did not repeat the full payload rehash.

This snapshot is complete for its eligible inventory, not for every empirical
file in the workspace. Its policy excludes files at least 2 GiB, AVI/MP4 videos,
outputs and transient caches. The manifest lists 1,730 excluded paths and
1,070,605,488,639 excluded file bytes. Symlinks record link targets, not the
contents at external targets. Keep separate protection for those omitted data.
The snapshot predates the review-follow-up changes; its stored Git commit is
not a claim that the current working tree is backed up.

New general snapshots use `ops/backup/snapshot.py`. Files **at least 2 GiB** are
excluded and recorded in the plan/manifest. Symlinks are preserved without
following their targets. Full-file SHA-256, expected-file coverage, mount/free
space checks and fresh-directory restores prevent incomplete backups from
claiming success. A default repository plan may still include substantial
eligible data: inspect it before creating a new full snapshot.

```bash
make backup-plan
make backup BACKUP_DEST=/mnt/hdd/backups/FiberphotometryCode
make verify-backup SNAPSHOT=/path/to/sealed/snapshot
make restore-backup SNAPSHOT=/path/to/sealed/snapshot RESTORE_DEST=/new/empty/location
```

These commands create no notifications. `--previous-snapshot` supports fully
verified immutable file references rather than source hardlinks. Keep referenced
ancestor snapshots. Legacy mirrors are coverage evidence; they cannot silently
become sealed content-verified snapshots.

To restore Git history separately, clone `repository.bundle` into a new directory.
Verify the bundle and SHA-256 values against `backup.json` first. A source archive
can also be extracted into a fresh directory and tested there. Existing source restore evidence is under `outputs/consolidation_20261021`.
Keep empirical recovery and code rollback separate.
