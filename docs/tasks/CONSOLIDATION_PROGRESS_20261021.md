# Consolidation execution state — 21 October 2026

## Resume here

Continue in `/home/fsp585/Desktop/FiberphotometryCode/.worktrees/legacy-consumers`, branch `codex/legacy-consumers`. This contains all current integration and caller-migration work. The primary checkout remains on codex/modular-refactor for the original viewer/library setup; do not switch/reset/clean it or the original delivery worktree. No merge is left unresolved.

## Captured and tested branches

| Branch | Role | Evidence |
| --- | --- | --- |
| codex/modular-refactor | Original source checkpoint31326cc, later continuation notes | Original source snapshot and preservation tag |
| codex/karen-seqnmf-delivery | Source checkpoint2c43b75 | 45 reviewed source/test/doc files preserved; original empirical outputs unchanged |
| codex/main-foundation | Foundation candidatef7c4056, based on origin/main and captured modular history | 850 CPU tests in lab and clean environments; architecture/lint/maps/package gates |
| codex/main-seqnmf | Combined candidate525b589 | 937 CPU tests in lab and clean environments; all gates; installed source-origin checks |
| codex/legacy-consumers | Current stacked caller-migration candidate | 951 CPU tests in lab and clean environments; all gates; exact extracted-statistic parity |

main remains67468c1 and no remote ref has changed. Scaffolding and legacy branches are already ancestors, not separate integrations still to apply.

## Completed work

- Source-only tar snapshots with per-file SHA256 and representative restore checks cover both original worktrees, including58 ignored potential source files in primary.
- Preservation tags codex/pre-main-20261021, codex/modular-preserved-20261021 and codex/seqnmf-preserved-20261021 restore from the verified703830120-byte all-refs Gitbundle.
- Recovered20 ignored photometry/language-model source files; exact numerical-definition/import parity retained. Narrow Git exceptions admit source while excluding payloads. Navigation now includes explicitly named source data packages without traversing empirical trees/symlinks.
- Declared tqdm/psutil in CPU test dependencies and pinned CI requirements. A temporary CPU environment without shared site packages passes the full suites and pip check; the shared lab environment was not upgraded.
- Merged preserved seqNMF source into a local candidate. Preserved complete delivery task history and regenerated conflict-affected maps.
- Verified installed fpcode modules and candidate study modules have matching candidate source hashes, with no modeling-backend initialization. All six functions recorded by the frozen atlas runtime manifest retain their original hashes across three modules. This is source provenance, not an empirical rerun.
- Migrated8 study launchers to canonical CLI modules with preserved configs/argv/failure status. Moved orchestration tests to canonical imports and kept explicit historical alias identity tests.
- Extracted the presentation Jensen-Shannon helper into fpcode.statistics.divergence;6 active consumers use it. Analytical bit-unit tests and exact float32/float64 parity use a new frozen pre-extraction reference. No existing numerical goldens or lint inventory changed.
- Added a hashed static legacy ledger with617 paths and recorded scan limitations. Review bundles/historical notebooks are classified separately from current source.

## Validation and recovery locations

Versioned evidence: `registry/checkpoints/consolidation_20261021.json`, `registry/SOURCE_RECOVERY_20261021.json`, `registry/LEGACY_*_20261021.*`, and `docs/architecture/LEGACY_RETIREMENT.md`.

Detailed logs, manifests, snapshots and recovery bundles are in the primary checkout's `outputs/consolidation_20261021`. Lab interpreter: `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`. Clean environment path is recorded in `clean_cpu_environment.json`; do not assume a temporary environment survives another machine/session. Numerical pools were limited to two threads. Six existing warnings remain. Final gates:951 CPU tests pass in both environments; architecture202modules/no violations or cycles; lint has no new debt; maps and outside-checkout package checks pass. Run relevant checks again only after relevant changes or for a new merge candidate.

## Outstanding gates

- Git fetch works; dry-run push fails because HTTPS GitHub credentials are unavailable. Remote PR publication/main promotion awaits an authenticated route; do not bypass protection or force push. Local candidates are reviewable meanwhile.
- Root core/pipelines/compat/archive are retained. Stored-object names, remaining project imports, external launchers and filesystem obligations need per-capability review before removal.
- Large empirical artifact backups and browser feedback exports are not fully verified. No data/model/media move, refit, new render or artifact conversion has occurred. Existing viewer processes were preserved.
- Local diagnostic/scratch/generated HTML files among the remaining ignored source candidates are preserved in the source snapshot; classification is still needed.

## Exact next step

Read docs/architecture/LEGACY_RETIREMENT.md and the legacy ledger. Migrate the next small set of active project imports with focused tests, then prove loader/identity support for named trusted artifacts before retiring an adapter. If GitHub authentication becomes available, publish the foundation, seqNMF and consumer candidates as sequential PRs (bases main, foundation, then seqNMF), preserve ancestry, require their checks, and promote in that order. Refresh remote refs before publishing. Do not independently merge the ancestor scaffolding/legacy branches.
