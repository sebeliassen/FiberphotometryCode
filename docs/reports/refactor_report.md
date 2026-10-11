# Refactor report: October 9–11, 2026

This report records the architecture and storage work and the evidence available
at the October 11 checkpoint. It was corrected after review to distinguish
measured results from requirements and to describe remaining compatibility work.
Earlier measurements below retain their original scope; they are not refreshed
metrics for every later working-tree change.

## Work completed at the original checkpoint

The reusable implementation was extracted into `src/fpcode`, with contracts,
alignment, tracking, photometry, behavior, KPMS, statistics, visualization,
workflows and installed CLI modules. A static dependency audit and small fixture
suite were added. Scientific policies that differed historically remained
explicit rather than being silently merged.

The migration inventoried 839 analysis Python files (208,854 lines), classified
reusable code, studies, verified redundant backups and vendor code, and recorded
moves and compatibility aliases in `registry`. The customized Keypoint-MoSeq
fork moved to `vendor/keypoint_moseq_syll`; verified redundant backups were
preserved under `archive/legacy`. Active study scripts were relocated, and legacy
runner imports were preserved with delegation shims.

Approximately 855 GB of existing empirical artifacts were relocated into a
`data` hierarchy. Historical code archives were offloaded to HDD during the disk
space emergency. The reported observation was 219 GB free on the NVMe SSD after
that work, compared with 3.4 GB beforehand; free space is a time-specific
observation, not a guaranteed current value.

## Recorded measurements and their scope

| Observation | Evidence and limit |
| --- | --- |
| 420 fast tests in approximately 6.2 seconds | Original local functional gate, using tiny fixtures; not a full empirical scientific reproduction |
| 121 library modules, zero audit violations/cycles | Original AST dependency audit under its configured rules; deliberate bridges and dynamic imports require separate scrutiny |
| 677 compatibility shims | Original registry count; preserving imports does not guarantee filesystem path compatibility |
| Duplicate copies reduced from 767 to 51 | Historical registry metrics at the original checkpoint |
| Files containing `/home/fsp585` reduced from 841 to 334 | Historical whole-workspace metric; hardcoded paths remained in studies, ops, tests and one library compatibility facade |
| `sys.path` occurrences reduced from 792 to 348 | Historical metric; occurrences remained in active study consumers as well as test/ops code |
| Two-session GPU diagnostic passed in 24.98 seconds | Two distinct sessions, 256 prefix frames each, exactly two AR and two SLDS updates on `cuda:0` |
| Sealed 96.37 GB eligible snapshot | 96,373,603,551 file bytes and 10,086 inventory entries; exclusions and Git bundle described below |

The functional suite validates selected extraction equivalence, independent
small numerical expectations, compatibility identity and other bounded
behaviors. The historical photometry fixture contains 256 rows per channel and
512 modeled frame indexes from one SessionRecord. Its provenance records two
original SessionRecords loading through retained namespaces. This is evidence
for those checked objects, not 100% compatibility for two years of pickles.

The GPU evidence is stored in
`outputs/20261011_142618-two-session-validation-7d681004/validation.json`.
It records actual device/update counts, finite arrays, checkpoint reload and
valid-frame evaluation. Source size/mtime/inode signatures stayed unchanged.
Four updates do not establish convergence, universal sampler equivalence,
bit-for-bit scientific invariance or full-cohort memory behavior. Those remain
requirements to verify with appropriate empirical comparisons. See
[validation guidance](../architecture/VALIDATION.md).

## Backup evidence

The sealed snapshot is
`/mnt/hdd/backups/FiberphotometryCode/20261011T143033Z-snapshot-353ab324`.
Its manifest records Git `e7010dd`, 8,440 files, 1,316 directories and 330
symlinks. A separately hashed Git history bundle contains 702,166,666 bytes.
Snapshot creation performs full-file SHA-256 verification before sealing and
publication. A later metadata-only review confirmed the manifest checksum and
read-only snapshot root without rehashing the large payload.

The snapshot excludes files at least 2 GiB, AVI/MP4, outputs and transient
caches. Its manifest lists 1,730 excluded paths and 1,070,605,488,639 excluded
file bytes. External symlink targets are not copied. This is recovery evidence
for the included inventory, not proof that every empirical file has independent
redundancy or that all data can be restored. No final source restore drill has
been recorded. See [recovery guidance](../recovery.md).

## Review corrections and current ownership

The original report's “production-ready,” universal “zero scientific regression,”
“100% pickle compatibility,” and “zero hardcoded paths” claims exceeded the
available evidence. The old `archive/compat` was a live dependency, organized
ops paths sometimes forwarded back to `ops/scripts`, and data still remained
inside study trees. A passing import test did not establish that old literal
`syllable_generation/data/...` paths resolved.

The review follow-up establishes project ownership under `studies/<project>`,
without top-level behavior/photometry study partitions. Shared mechanisms remain
in `src/fpcode`; project methods and cohort policy stay with their project.
Operational implementations live under purpose-specific `ops` directories;
`ops/scripts` preserves caller aliases. Live historical imports belong in
`compat`, while `archive` holds historical material and frozen bundles.
Empirical inputs belong in `data`, including project recordings in `data/raw`.
Each relocation needs an import audit and filesystem-reference audit.

Current boundaries and test selection are defined by the maintained
[capability contract](../architecture/UNIT_CONTRACT.md), `pyproject.toml` and
architecture rules, rather than the original “clean 12-directory root” count.
The installed CLI is canonical; historical `pipelines` names are compatibility
entry points. Later validation and metrics must be reported with the working
tree or commit they checked.

The code-review ZIP intentionally adds a root `REFACTOR_REPORT.md` copy of this
file for reviewers. It is an archive export, not a second maintained report in
the repository. Historical architecture/runbook documents remain preserved and
are identified as dated references in the [documentation index](../README.md).

## Remaining validation

Representative historical session reconstruction and numerical comparisons
should expand as active consumers are migrated. Full-cohort backend behavior and
scientific convergence require dedicated validation. Snapshot restoration needs
a fresh-directory drill, and large excluded data need separate recovery evidence.
Compatibility retirement must follow a consumer inventory, rather than deleting
aliases merely because a fixture suite passes.
