# Consolidation and gradual integration into main

Prepared 21 October 2026. Status: execution authorised and started; see CONSOLIDATION_PROGRESS_20261021.md for the current checkpoint.

## Objective

Make the current framework and Karen seqNMF work available on main, then progressively remove dependencies on root core, pipelines, compat and archive. Each integration must leave main usable and each retirement must have evidence that active consumers and required artifacts still work. Scientific changes, infrastructure integration and physical data relocation are separate changes.

The target is one authoritative implementation per capability in src/fpcode, project-owned scientific policy in studies, and supported command-line entry points in fpcode.cli. Active source, configurations, tests and documentation belong in Git. Experimental recordings, model payloads, videos and generated run products remain separately managed empirical artifacts.

Do not require every exploratory analysis to be scientifically accepted before its source can be preserved on main. Label exploratory workflows and their limits accurately; do require the documented software behaviour to pass its checks.

## Verified starting point

Local observations, not a refreshed account of GitHub:

| Item | Observed state |
| --- | --- |
| main | 67468c1, 8 October 2025 |
| scaffolding-refactor | 0891de1, ancestor of legacy and modular |
| codex/legacy-consolidation | b83298f, ancestor of modular |
| codex/modular-refactor | 15df93d, primary checkout; 61 commits ahead of main, with no main-only commits in these local refs |
| codex/karen-seqnmf-delivery | 736f3ca, separate worktree; branches from 6c179ae; eight delivery-only and two modular-only commits relative to these tips |
| Primary working changes before this plan | Modified docs/tasks/CURRENT.md and an untracked handoff presentation |
| Delivery working changes | Modified source/configuration/tests and substantial untracked theme, measurement and gallery source; branch tip alone is incomplete |
| Root core | 15 tracked Python forwarding modules |
| Root pipelines | Five tracked Python entry points and one generated map |
| Root compat | 1,095 tracked entries, including 583 Python files; classification still needed |
| Root archive | Only ARCHIVE_INDEX.md and the compat symlink are tracked; most local material is ignored |
| Archive links | archive/compat points to ../compat; archive/legacy points to existing external storage on /mnt/hdd |
| Environment | Read-only CPU doctor passes with Python 3.10.19; installed distribution metadata says 0.1.0 while checkout source says 0.2.0 |

Concrete dependencies found: multiview study launchers invoke pipelines/train.py and pipelines/fuse.py; integration tests import/patch pipelines modules; fresh-process tests verify core aliases. Several language-model scripts also use a bare pipelines.presentation namespace, requiring ownership-aware migration rather than text replacement. An empirical test references an archive/legacy checkpoint.

Recent Karen outputs used study code from the delivery worktree and shared fpcode imports from the primary checkout. Preserve the actual runtime source manifests; a clean delivery commit alone does not reproduce those runs. Delivery reference_seqnmf.py and shared behavior/episodes/seqnmf.py implement different update policies and must not be unified as a mechanical cleanup.

## Git strategy

1. Preserve work on its existing branches first. Make explicit source-only checkpoint commits after reviewing file lists, then tag the captured primary, delivery and main baselines. A tag protects committed content, not dirty or ignored files.
2. Refresh remote information before implementation. Fetch without checking out or altering either working tree; re-evaluate ancestry and any existing PR. Do not assume the local origin refs observed here are current.
3. Create a separate integration worktree from the verified main base, on a codex/ branch such as codex/main-foundation. Keep the primary and delivery worktrees available for comparison and existing viewers.
4. Bring modular history into main first, with current compatibility retained. Scaffolding and legacy are already ancestors: do not merge or cherry-pick them independently.
5. Bring the captured delivery history into the resulting main in a second integration PR. Merge conflicts in task notes must preserve dated evidence while providing a clear newest state.
6. Branch each later cleanup from updated main. Use one coherent responsibility per PR. When dependent PRs overlap, stack them explicitly and retest against the actual combined candidate.
7. Preserve ancestry with normal merge commits for the historical integration PRs. Avoid rebasing or force-pushing existing research branches. Later small cleanup PRs may follow the repository's chosen merge policy.
8. Keep main and previous release tags recoverable. Retain research branches until all unique commits are reachable from main or deliberately retained as documented historical work. A merged branch is not a backup of empirical artifacts.

The first foundation PR will be substantial because main is a year behind. Review it by capability, mechanical moves and scientific policy, using the existing commit history. The intermediate 3ec30fb is a possible review boundary, but not automatically a release candidate: it lacks newer CPU guard/navigation infrastructure. If the foundation needs splitting, require working validation at each split; do not promote an old checkpoint simply because its name sounds stable.

After each successful merge, record the main commit, tested interpreter/environment, checks, support status and rollback reference. Test the current PR head or synthetic merge result, then verify the actual merged tree. Do not rely on checks from an earlier branch tip. Retest after relevant conflict resolution.

## Phase 0 — Preserve and describe the actual current state

Deliverable: recoverable source checkpoints and an artifact/dependency inventory. No legacy removals.

- Record both worktree statuses, tracked changes, untracked files, relevant ignored maintained source, branch SHAs and known running jobs/viewers. Do not use a blanket git add, clean, reset or stash to achieve a tidy tree.
- Review and checkpoint primary notes/presentation and delivery implementation in coherent commits, retaining earlier outputs and human feedback. Keep recordings, models, caches and generated media out of those commits.
- Preserve a source snapshot and Git bundle containing all needed refs, with hashes and a small restore exercise into a fresh destination. Review existing backup policy: it excludes outputs, video and analysis caches, and caps file size; it does not establish complete empirical recovery.
- Inventory empirical dependencies separately: canonical path, symlink target, size, existing hash or planned streamed hash, owning run and backup status. Copy only missing required artifacts under an explicit storage budget; verify before any later move. Do not follow an external symlink recursively by default.
- Export relevant browser-local feedback JSON. Identify environment/console scripts or scheduled jobs using old physical paths without launching them.
- Store runtime source manifests, fit recipes, native mappings, masks, source hashes and frozen output identities alongside the source checkpoints.

Exit gate: committed source is recoverable, remaining dirty/ignored exceptions are documented, and required artifacts have known locations and recovery coverage. Missing coverage is recorded as a dependency for the affected later move, rather than concealed by a clean Git status.

## Phase 1 — Integrate the existing modular foundation

Suggested PR: codex/main-foundation -> main.

- Merge the pinned, captured modular branch into the isolated main-based candidate. Keep core, pipelines, compat and archive entry points operational.
- Review accumulated source additions for accidental empirical payloads, bundled duplicates, generated products and environment-specific paths. Put only required repairs into the foundation candidate; record unresolved historical material explicitly.
- Fix integration or installation defects without redesigning numerical algorithms. Preserve API identities, calibration resources and historical defaults behind their documented interfaces.
- Document supported commands, dependency environments and exploratory studies. Check that user-facing examples point to real supported workflows.
- Run the full CPU functional, architecture, lint, navigation and outside-checkout package checks. Run focused photometry, fusion, training lifecycle and compatibility regressions as appropriate.
- Verify available CI and branch-protection configuration on GitHub. Existing workflow files are evidence of configuration, not evidence that remote checks passed.

Exit gate: the merged foundation is installable and its documented CPU paths pass from the integrated revision. Main can now be the base for new work even while compatibility remains. A minimal supported CPU installation needs its own dependency check: the current package-check reuses lab scientific dependencies and does not prove a clean dependency installation.

## Phase 2 — Integrate the preserved Karen delivery

Suggested PR: codex/main-seqnmf -> main, after phase 1.

- Merge the captured delivery branch into updated main, including the currently untracked source/tests after phase 0. Preserve earlier fit and catalogue provenance and distinguish superseded from current notes.
- Remove accidental dependence on another editable checkout in the validation environment. Install the candidate wheel in a temporary environment, run study drivers from the candidate worktree, remove inherited PYTHONPATH/root overrides, and assert actual shared module origins. Use explicit code/data roots rather than global sys.path surgery.
- Preserve the recovered-reference solver separately from the shared alternative. Keep group-blind definitions, handling masks, missing-frame barriers, unknown labels, exposure and native half-open intervals unchanged.
- Run delivery unit/regression tests and the full integration gates. Confirm existing recipe/source/output hashes before comparing results.
- Define an optional empirical acceptance pass separately: inspect the frozen 99-recording/21-mouse/7,319-core identity tables without refitting, compare one bounded representative theme measurement and a short native video witness, and check a gallery's playback plus feedback import/export. Use a new output location and explicit inputs/resources. Agree the exact execution scope before launching it; this planning task does not authorise a new fit or render.

Exit gate: one captured integrated revision supplies both study and library code, and frozen outputs remain traceable. Exploratory statistical and behavioural acceptance limits remain visible. Baseline old results are not silently replaced by newly generated ones.

## Phase 3 — Build the retirement ledger and migrate active callers

Suggested first cleanup PR: codex/legacy-consumer-inventory -> main.

For each candidate module/path, record: current location, authoritative target, responsibility, active callers, physical launch paths, stored-object names, config/resource references, required tests, artifact evidence, support decision and retirement gate. Keep the ledger with registry migration records.

Classification must distinguish shared algorithm, study policy, CLI adapter, historical serializer contract and inactive material. Search tracked Python, shell/configs, notebook code cells, dynamically constructed imports, __file__ assumptions, documentation and known external operational consumers. Static search is evidence, not proof of no external use. Record unknown consumers explicitly and publish a migration/release note for external users.

Migrate users to canonical imports and supported CLI/module invocations while keeping aliases temporarily. Move reusable mechanisms into capabilities and scientific settings into studies/configuration. Preserve exact numerical policies during relocation. Correct bare pipelines.presentation imports to the actual project owner; do not route them into the top-level command-line adapter package.

Exit gate: the ledger identifies supported consumers and each changed call site has relevant validation. New code is prevented from introducing retired dependencies by an architecture/navigation check that allows only specifically documented compatibility exceptions.

## Phase 4 — Retire core and pipelines incrementally

Suggested PRs: codex/retire-core, then codex/retire-pipelines, each from updated main.

- Prove current consumers invoke fpcode capabilities or installed/module CLI entry points. Include study launchers, notebooks and operational commands.
- During transition retain fresh-process identity checks for both import orders, direct scripts, runpy and file/spec loading where those are supported. Tests that patch old pipeline names must move their patch targets with their production imports.
- Exercise supported CLI success, expected failure, no fake SUCCESS records and harmless help/dry-run behaviour. Inspect each command's dry-run support; do not assume it exists. Test out-of-checkout execution with a valid tiny fixture.
- Remove each root adapter only after its support decision is explicit. If saved objects require its old name, defer that adapter or provide a separately tested compatibility route. Do not add an import hook as a substitute for promised physical script behaviour.
- Preserve meaningful tests: canonical command behaviour and numerical expectations continue. Remove obsolete alias assertions only as part of the documented support retirement, with consumer/artifact evidence; do not delete failing tests to manufacture a pass.
- Update shim generation tooling as well as SHIMS.tsv, or it may recreate the removed files. Update Makefile, CI, maps, docs and package tests where affected.

Exit gate per PR: supported consumers and artifact routes pass without that root directory. No undeclared filesystem entry point remains required.

## Phase 5 — Reduce root compat by capability

Suggested PR series: photometry/session loading, tracking/fusion, KPMS, then visualisation/sequence consumers; reorder based on the inventory's dependency graph.

- Extract any remaining useful mechanism once, with explicit inputs/contracts and independently checked expectations. Complete missing generic session-source/movie-composition boundaries only where a concrete consumer needs them; do not create a parallel framework by copying a historical module.
- Keep root compat distinct from src/fpcode/compat. The latter contains historical policies and packaged calibration resources: removal of the root directory does not imply deletion of those resources or replacement of their numerical policy.
- For each required stored-object family, choose a supported route: load through canonical classes if that works, convert a trusted source artifact to an explicit versioned representation, or retain a small installable legacy loader component. Prove the route on identified artifacts; do not claim universal pickle support.
- Converters must write a separate destination, preserve originals, check schema/units/clocks/masks/identity, and be safe to rerun or refuse ambiguous overwrites. Never bulk-rewrite empirical pickles as routine cleanup. Inventory payloads without loading arbitrary pickles.
- Test candidate installations without the root compat directory on the search path. Remove its pytest path injection only after the supported functional suite and migrated study paths pass without it.
- Remove aliases in audited batches. A final root compat retirement waits for every supported family; a deliberately retained small component is acceptable if migration cost or historical replay requires it.

Exit gate per capability: a working current path plus a documented historical recovery path, independently checked behaviour and no active undisclosed dependency on the removed namespace.

## Phase 6 — Externalise archive safely

Suggested PR: codex/externalise-archive, followed by a separately scoped storage operation if needed.

- First remove active code/test/config dependencies on archive/compat and archive/legacy. Required small golden fixtures stay versioned; large empirical validation inputs move behind explicit input configuration.
- Audit archive content, symlink targets, historical baseline references, source manifests and old scripts that depend on its location. Preserve a Git/source reference and retrieval index, including source/run hashes.
- Select external storage with capacity and recovery coverage. For local content requiring relocation, copy and verify before retiring the old location. Keep external symlink targets intact; removing a link is not authority to remove its target. Stream hashing/copying with resource limits.
- Separate changes to the two tracked archive entries from physical ignored-file changes. A PR merge does not migrate ignored files on any workstation. Provide and validate an explicit local migration procedure, including a restore into a new directory.
- Keep the historical catalogue/retrieval instructions in docs or registry. Old dated reports can retain historically accurate paths with a current location crosswalk; do not rewrite their scientific provenance to imply a different original run.

Exit gate: active package/tests/studies work without root archive, required historical artifacts can be retrieved and verified, and the local storage transition has independent recovery evidence. The old directory can remain locally until that gate passes even after active source no longer needs it.

## Validation matrix

| Change | Focused evidence before merging |
| --- | --- |
| Import/path relocation | Fresh processes, both import orders and object identity while supported; physical/direct/module/installed invocation; literal and constructed path audit |
| Photometry/session mechanisms | Existing independent historical-session slice expectations; channel/time alignment, endpoints, masks, missing values and immutability |
| Tracking/fusion | Camera correspondence, confidence masks, native clocks, calibration resources and fixture fusion results |
| KPMS/runtime/compaction | Backend-free imports/help, patch restoration on success/failure, checkpoint contracts and numerical fixtures; bounded selected GPU evidence only when runtime behaviour changes |
| SeqNMF/themes | Solver policy, seeded fixture outputs, gap barriers, native call endpoints, stage conservation, exposure, ranking and correction calculations |
| Movies/review galleries | Native frame mapping, timing/panel identity, source pixels, no missing/dark unintended panels; playback and feedback persistence/import/export |
| Packaging | Wheel/resource check outside checkout; assert import origin, clear inherited path/root overrides, installed CLI help and a meaningful tiny workflow |
| Historical conversion | Trusted representative input, unchanged source hash, schema/identity/clock parity after reload, explicit failure cases and separate outputs |
| Physical archive relocation | Inventory/manifest, no-follow symlink policy, streamed copy/hash verification, updated references and a restore/retrieval exercise |

For shared-behaviour milestones run the complete functional suite plus architecture/lint; package and navigation changes also require their gates. Preserve numerical goldens and independent identity assertions. Do not refresh accepted lint debt or regenerate expected values to absorb unexplained failures.

Use focused checks during development and one broader pass on each completed merge candidate. Repeat a passed check after relevant code/conflict changes, a failure or a remaining concern. Do not rerun full suites after every documentation edit.

Current commands, with a deliberately chosen interpreter:

```bash
export PYTHONDONTWRITEBYTECODE=1
FP_MIGRATION_PYTHON=/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python
"$FP_MIGRATION_PYTHON" -B -m fpcode.cli.doctor --json --require cpu
make test-cpu PYTHON="$FP_MIGRATION_PYTHON"
make architecture PYTHON="$FP_MIGRATION_PYTHON"
make lint PYTHON="$FP_MIGRATION_PYTHON"
make folder-maps-check PYTHON="$FP_MIGRATION_PYTHON"
make package-check PYTHON="$FP_MIGRATION_PYTHON"
```

Regenerate maps after relevant source/config changes and review the diff before checking. Run CPU CI in its pinned environment and check one minimal supported dependency installation in an isolated temporary environment; do not modify the shared lab environment incidentally. Current wheel tests reuse available scientific dependencies, and passing doctor metadata does not prove GPU readiness or scientific equivalence.

Any substantial empirical check needs named inputs/output, expected duration, CPU/GPU/thread/RAM settings, disk budget and stopping criteria. No full-cohort fit/search, compaction, notification, broad video render or bulk data move is an automatic regression check. Failed tests are investigated before progressing; discrepancies receive numerical/scientific explanations rather than relaxed tolerances without evidence.

## Rollback and completion

- Before each promotion, retain the last known working main commit and source/artifact manifests. Use reviewed revert commits for shared main; do not rewrite history with a force push.
- Separate retirement from numerical changes so an adapter/caller change can be reverted without changing analysis definitions. A merge revert uses its recorded first-parent reference and must be tested; reintroducing a reverted branch later needs an explicit plan because ancestry remains merged.
- If a physical move fails, use the verified copy/restore plan. Git reverts cannot restore ignored recordings, external symlink targets or browser-local feedback.
- Keep old outputs immutable and write checks to new locations. Compare exact identities and known tolerances; artifact generation timestamps alone are not parity evidence.
- Record only the evidence obtained: fixture/import tests, bounded empirical comparisons and full-population frozen-table audits prove different things. A preserved small legacy loader is preferable to unsupported recovery claims.

Complete when main contains the captured intended active work; supported current workflows run from its candidate installation; old root directories are absent or have an explicitly justified temporary obligation; artifact recovery is proven for named retained inputs; and documentation names remaining limitations and maintainers.

## Immediate next step

Execute phase 0 only: prepare reviewed source checkpoints and the dependency/recovery inventory in both existing worktrees, then review the foundation PR candidate. No merge into main, legacy deletion, empirical workload or physical archive move is performed by this planning task.

## Checks performed while writing this plan

Read-only branch/merge-base/status inspections, tracked-file and shallow symlink inventories, selected caller/test/package/backup-policy inspection, and CPU doctor with /home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python passed. No fetch, commit, tag, branch creation, merge, remote update, functional suite, model execution or artifact relocation was performed. Only this plan and an additive task-state entry were written.
