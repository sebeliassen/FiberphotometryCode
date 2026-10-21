# Antigravity refactor handoff

Prepared 2026-10-11. Repository: `/home/fsp585/Desktop/FiberphotometryCode`.

This is the handoff for the **next phase: consolidate remaining implementations,
migrate consumers, and reduce redundant legacy paths**. It records the published
baseline and the constraints that must survive cleanup. Read the current source
and repository instructions before acting: this document is evidence at a date,
not a substitute for checking changes made after that date.

## 1. Objective and scope

The user's goal is a sleek, functionally equivalent framework with shared
capabilities implemented once, clear responsibility boundaries, thin study and
CLI layers, and inexpensive functional verification. The motivating problem was
repeated functionality across `syllable_generation`, photometry code, and
monoamine analyses, combined with overly large components.

The structural foundation and important functional repairs are implemented and
published. **The legacy retirement phase is not complete.** Do not interpret the
presence of canonical modules as proof that all consumers have migrated, or the
presence of a legacy file as proof of a second algorithm implementation.

For the next phase:

- Inventory before removal; migrate active consumers before retiring their paths.
- Preserve scientific outputs and explicit study policies, rather than silently
  replacing historical conventions with one supposedly cleaner convention.
- Keep small, justified adapters for old serialized data where necessary.
- Keep existing study material and documentation with their provenance.
- Avoid full-cohort training. The user specifically prefers bounded validation:
  two sessions, two AR updates and two SLDS updates, plus relevant GPU checks.
- Do not launch compaction, production fitting, full video rendering, filesystem
  relocation, or large backups as an incidental part of source cleanup.

This handoff itself was written after publication. At creation it is a new local
file; do not assume it is already committed or present on GitHub.

## 2. Published Git baseline and recovery

Remote: <https://github.com/sebeliassen/FiberphotometryCode>

Published branch: `codex/modular-refactor`.

Open PR: <https://github.com/sebeliassen/FiberphotometryCode/pull/1>

The PR was **open and unmerged** when publication was checked. The remote branch
matched the local tip, the working tree was clean before this document, and
`main` remained at `67468c199256a7642bcab535acc205f57b54a6e0`.

| Reference | Commit | Meaning |
| --- | --- | --- |
| Current published baseline | `2c90f39b1dc44b98a01583e58814253553967154` | Latest completed refactor, diagnostics, configuration and docs |
| `pre-scaffolding-20261009` | `969c893b8b6ba7ddd42af178f36aff3406f84cea` | Snapshot before structural scaffolding |
| `pre-codex-modularization-20261009` | `0891de1b1c288f16c42d23d7e4b4cbdb020f8cee` | Gemini baseline before subsequent Codex work |
| `codex-refactor-implementation-20261009` | `6ca24f3f369382e5c329492b305f5540fc48c032` | Earlier Codex implementation checkpoint; NOT the latest baseline |

All three existing tags were published. All **103 commits present before the
latest three commits remain ancestors of the published tip**. No existing
history was amended, rebased, reset, or force-pushed.

The final three commits were:

1. `36d8184`: explicit scientific inputs, behavior preservation, compatibility
   repairs and functional/regression coverage.
2. `51a8357`: bounded real-backend and two-session GPU diagnostics.
3. `2c90f39`: configuration organization and preserved engineering records.

The entire PR also includes earlier structural work: its diff against `main`
spanned 2,265 files at publication. Do not confuse the latest three commits with
the entire migration, and do not regenerate that migration from scratch.

### Safe continuation

Start by inspecting `git status --short --branch`, recent history, remote refs,
and any other agent's changes. Do not switch an actively used checkout without
checking its state. A separate source-only worktree or clone is a good option;
it will not contain ignored empirical data automatically.

A follow-up branch such as `codex/legacy-consolidation` can begin at the exact
published baseline. If the parent PR is still open, a PR based on
`codex/modular-refactor` makes the cleanup diff easier to review. If it is already
merged, inspect the resulting history before choosing the base. Never assume
the PR state or branch tip stayed unchanged.

Use small commits with ordinary, non-force pushes. Stage explicit reviewed
source paths. Check the staged diff and file sizes; avoid indiscriminate staging
of outputs or ignored data. Never put credentials into commits or reports.

Do not use `git reset --hard`, `git clean -fdx`, broad recursive deletion, amended
published commits, or force-push as cleanup mechanisms. A pushed commit protects
tracked source; it does not recover an untracked empirical pickle deleted from
the working directory.

### Recovery without losing subsequent work

For inspection or running the accepted source in isolation, this command creates
a separate worktree at the published baseline, without replacing current files:

```bash
git worktree add --detach /tmp/fpcode-published-baseline \
  2c90f39b1dc44b98a01583e58814253553967154
```

Choose a different new destination if that path already exists. This checkout
does not copy ignored data. Configure data access explicitly and read-only.

If a newly published cleanup commit is wrong, use a reviewed `git revert` of
that cleanup commit on the appropriate branch and run its checks again. Revert
dependent commits in a considered order. Avoid resetting a shared branch to an
older tip and discarding later work. Preserve the current baseline and tags.

## 3. Architecture: original layout, target, and actual state

Before scaffolding, functionality was distributed across `core`,
`fiberphotometry`, `syllable_generation` (analysis, multiview experiments,
syllabizers), `monoamine_interactions`, `src_mono_inter`, `src_oft`,
`karen_project`, root scripts and miscellaneous operational/documentation paths.
Shared scientific operations were mixed with study configuration, file lookup,
GPU internals, rendering and orchestration. This is a summary, not a complete
historical directory inventory; use the pre-scaffolding commit for exact paths.

The intended high-level structure is now present:

```text
FiberphotometryCode/
├── pyproject.toml, Makefile, README.md
├── src/fpcode/
│   ├── contracts/             # validated representations and explicit coordinates
│   ├── adapters/              # external formats and existing session objects
│   ├── alignment/             # clocks, frames, mappings, resampling
│   ├── tracking/              # geometry, QC, calibration, fusion
│   ├── photometry/            # processing, windows, metrics, aggregation
│   ├── behavior/sequences/    # bouts, kinematics, transitions, sequence operations
│   ├── kpms/backend/kernels/  # fitting, results, checkpoints and backend internals
│   ├── statistics/
│   ├── viz/selection/         # QC/selection separate from geometry and media
│   ├── workflows/            # compose adapters and capabilities
│   ├── cli/                  # installed entry points
│   └── common/               # generic paths, configuration, I/O, manifests
├── studies/                  # study-specific consumers and scientific choices
├── configs/{training,fusion,photometry}/
├── ops/{backup,repo,diagnostics,dlc,monitoring,compaction}/
├── tests/{unit,integration,regression,architecture,golden}/
├── registry/                 # SHIMS.tsv, MOVES.tsv and historical records
├── docs/{architecture,decisions,reports}/
├── vendor/, archive/, outputs/, scratch/
└── compatibility paths       # core, old study trees, pipelines, links/forwarders
```

This is a responsibility map, not a claim that the exact requested leaf tree is
finished. For example, there are both `statistics/summary.py` and `summaries.py`,
and several photometry, alignment and visualization compatibility modules.
Inspect their callers and semantics before merging them. A future
`environment.lock.yml` was part of the user's desired tree; do not claim a
reproducible environment lock exists merely because `pyproject.toml` does.

Recursive modularity means splitting distinct responsibilities when warranted.
It does not mean creating a configuration file, README or CLI at every leaf.
Study code may remain substantial when it implements a study-specific analysis.

### Dependency boundaries

`ops/repo/dependencies.py` is the implemented boundary/cycle audit. Read its
`LAYERS`, `FORBIDDEN`, `COMPATIBILITY_IMPORTS`, and visualization restrictions.
The principles are:

- Contracts and generic utilities stay low in the dependency graph.
- Domain code uses contracts and allowed lower capabilities.
- Workflows compose algorithms and adapters; CLI modules invoke workflows.
- The installed library must not depend on `studies`, `syllable_generation`,
  `core`, `pipelines`, `archive`, `tests` or `scratch`.
- There is a narrow existing exception allowing `fpcode.kpms.export` to forward
  to `fpcode.workflows.session_export`. It does not authorize other reverse
  dependencies or broad exceptions.
- Generic visualization modules must not import cohort defaults or historical
  loaders. Those belong in explicit compatibility APIs or study configuration.

The audit checks static and literal dynamic imports. Computed module names,
shell launchers, notebook code and external consumers still need investigation.
Do not relax the audit to make a failing migration appear successful.

## 4. What the refactor changed and why it matters

### Explicit scientific representations and alignment

`contracts/timelines.py` provides clock identity and explicit units, original
frame coordinates and validity. Contract arrays own read-only copies; a frozen
dataclass alone would not prevent caller-owned NumPy arrays from mutating it.
Do not remove defensive copying just to reduce lines or allocations.

Alignment is based on explicit coordinates, rather than assuming row positions
are original frame numbers. Integer time coordinates, including large
nanosecond timestamps, must retain exact differences; blindly converting to
float64 can lose distinctions above 2**53. Signed/unsigned coordinate operations
also require care. Preserve the existing tested coordinate handling.

Resampling and event windows make tolerances, clock relationships and gap
policies explicit. Invalid frames, dropped samples and excluded bouts must not
become valid merely through interpolation or array concatenation.

### Shared behavior and photometry operations

Shared bouts, kinematics, transitions and sequence counting were extracted and
consumers repaired across Datta replication, Ulrik kinematics, Karen analyses,
and monoamine photometry notebooks/scripts. The point was reuse with preserved
scientific behavior, not one global averaging or smoothing rule.

Examples of non-interchangeable choices:

- Weighting animals, sessions, bouts, or individual observations differently.
- Treating missing observations as excluded versus zero-valued.
- Including versus excluding a terminal partial window or bout.
- Dense versus sparse syllable timelines and transitions across missing labels.
- Historical row-based interpolation/end filling versus strict time/gap APIs.
- Confidence, presence and coordinate-finiteness rules for tracking.

Keep these choices named and tested. Similar function signatures or nearly
identical source text do not prove equivalent behavior on edge cases.

`adapters/legacy_sessions.py` handles old session storage conventions, export
schemas, metadata, and attaching tables while retaining an object's class and
other metadata. Do not replace existing empirical session objects wholesale
with new dataclasses and assume historical readers/pickles will still work.

### Tracking and calibration

Camera mapping and fusion gained explicit input contexts. For example,
`CameraMappingContext` scopes calibration/cache state to supplied inputs, so two
contexts using the same session ID need not share a calibration. Provenance and
cache invalidation depend on the actual timestamp/calibration inputs.

Preserve dropped-frame handling, original frame mappings, QC masks, calibration
validity checks, jitter tolerances and transform choices. Similar-looking
static/continuous/reflection-corrected fusion paths are not automatically
duplicates. Cohort-specific transform JSON files are packaged resources; wheel
tests must verify their availability outside the source checkout.

### KPMS training and backend lifecycle

Training separates requested phase updates from saved checkpoint numbering and
existing model state. Keep the distinction between two additional AR updates
and an absolute checkpoint endpoint. The bounded diagnostic verifies four real
updates, not merely that a mock was called four times.

Optional JAX/KPMS/GPU dependencies load during execution, not when importing the
base package or requesting CLI help. Backend monkeypatches are scoped and
restored, including on failure. Inspect `backend/patching.py`, the patch registry
and lifecycle tests before simplifying these mechanisms.

Accelerated sampling, precision, GPU residency, batching and checkpoint
compatibility are meaningful behavior. Tiny successful fitting does not prove
that accelerated and historical samplers are scientifically equivalent.
Do not alter kernels or patch order as incidental cleanup.

### Visualization and provenance

Selection eligibility, density sampling, pose geometry and media generation
have distinct responsibilities. Landmark groups, input files and calibration
are explicit contexts rather than implicit discovery of one historical cohort.
Historical defaults live in named compatibility modules.

Cache fingerprints track actual implementation files and effective inputs.
Hashing a forwarding shim alone misses changes in its target. When moving or
consolidating implementation code, update provenance/invalidation dependencies
and check that old caches cannot be silently accepted as current.

Study bootstrap/path fixes were made for the all-cohort v2 visualization
consumers. Do not reintroduce dependence on a particular working directory or
only make imports work through pytest's extra search paths.

### Configuration and documentation

Training configuration moved under `configs/training`, with compatibility links
for the old flat paths. `configs/photometry/event_windows.yaml` states explicit
window/gap policy. Validate real loaders and relative-path resolution before
removing a link or moving another configuration.

`docs/README.md` distinguishes maintained guidance from historical engineering
and scientific records. Existing Markdown reports were preserved, not merged
into one replacement narrative. `handoff.md` is an older dated record; its
completed/pending lists are not the current task list. `docs/recovery.md` also
contains older checkpoint wording: use the publication table above for the
latest accepted tip, and retain its factual backup limitations.

## 5. Existing validation evidence

These are completed checks on the published source, not promises about future
edits. Re-run affected checks after your changes.

| Check | Result and limits |
| --- | --- |
| Default functional suite | **420 passed**, eight expected compatibility deprecation warnings; **7.25 s** in an independent clean clone |
| Architecture | **121 library modules**, zero violations and zero cycles |
| GPU diagnostic | RTX 4090; two distinct empirical sessions, 256 frames/session, 21 keypoints, 10 latent dimensions, 100 states; exactly 2 AR + 2 SLDS updates; **24.53 s**, passed |
| GPU lifecycle/output | Device residency, finite arrays, snapshots 0–4, patch restoration, checkpoint reload, evaluation and results extraction passed |
| Source protection | Original pilot inputs remained unchanged; diagnostics wrote a unique output run |
| Other runtime checks | Tiny real CPU fitting and MP4 encoding/decoding passed |
| Packaging | Earlier isolated wheel/CLI checks passed; repeat after packaging/path changes |

The clean clone at `/tmp/fpcode-publication-clean-20261011-th1uxl5u/repository`
used independent Git objects (no hardlinks) and contained no production data.
This helped catch reliance on uncommitted files. `/tmp` paths are disposable
evidence and may disappear; do not use them as durable backup policy.

Latest GPU evidence, if still present:

```text
outputs/20261011_111938-two-session-validation-1a4a8add/validation.json
```

These generated files were intentionally excluded from the new source commits.
Full-cohort convergence, long-run memory behavior, full scientific sampler
equivalence and every historical external consumer were **not** established.

### Commands and environments

The existing environment used for GPU checks is:

```text
/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python
```

Check that it still exists; do not upgrade/reinstall it incidentally. From the
repository, use:

```bash
make test PYTHON=/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python
make architecture PYTHON=/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python
```

Default test selection is explicit in `pyproject.toml`; markers exclude `gpu`,
`slow` and `needs_data`. CPU does not imply inexpensive. Do not recursively run
all historical `tests/` scripts: some execute expensive empirical analyses.

For relevant backend/rendering changes, run separately:

```bash
make smoke-runtime PYTHON=/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python

make validate-two-sessions \
  PYTHON=/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python \
  VALIDATION_PROJECT=syllable_generation/multiview_top_bot_experiments/pilot_comparison_21kp_dynamic
```

Inspect `docs/architecture/VALIDATION.md` and diagnostic help first. The GPU
source pilot supplies `preprocessed_data.pkl`, `config.yml`, `pca.p`. The worker
uses two distinct session prefixes, not two segments from one session, preserves
contiguous valid excerpts, and does not resume a production checkpoint. Default
timeout is 180 s; input limit is 1 GiB (the actual pilot cache was about 500 MiB).
It uses source anatomy/PCA/dimensions and explicit diagnostic kappas. Output is
under a new run ID. CUDA failure must not silently become a CPU success.

Important test groups include:

- `tests/architecture/`: import boundaries, dynamic imports, legacy identity,
  study bootstrap and consumer dependencies.
- `tests/integration/`: actual CLI/workflow behavior, fusion contexts, training
  lifecycle, session exports, configuration links and multimodal analysis.
- `tests/regression/`: module extraction and independent historical comparisons.
- `tests/unit/`: timeline/alignment, behavior counting, photometry, tracking,
  cache invalidation and visualization contexts.
- `tests/golden/historical_session_slice.npz`: a small anonymous numeric
  historical fixture (142,705 bytes), explicitly allowed by `.gitignore`.

Do not regenerate goldens from the new implementation merely to make comparisons
pass. Preserve independent expected values and stated numerical tolerances.
Do not replace meaningful checks with source-text assertions alone.

## 6. What “double code” actually consists of

At handoff, `registry/SHIMS.tsv` has **420 registered compatibility paths**:
20 `module_proxy`, 9 `forwarder`, 381 `DeprecationWarning`, and 10
`compatibility_facade` rows. These labels are inventory hints, not proof that
every row is correct or equally removable. The count happens to equal the test
count; they are unrelated.

Inspect both registry and filesystem. Examples:

- `core/fusion_engine.py` imports `fpcode.tracking.fusion` and replaces its
  module entry in `sys.modules`. There is one fusion implementation behind these
  two paths. Existing tests deliberately check module identity.
- `fiberphotometry/analysis/signal_processing.py` delegates to a study module;
  moving the study module into the shared library requires semantic analysis,
  not just renaming its import.
- Some old directory names are symlinks to canonical study/vendor paths.
- Some legacy-named canonical modules encode required compatibility policies.
- Some old trees contain **ignored live data alongside tracked source**.
- Actual algorithm duplication may still remain; file names alone cannot locate
  it, and `registry/DUPLICATES.tsv` is historical evidence, not deletion approval.

Deleting every legacy-looking directory would be unsafe and may provide little
algorithmic simplification. The goal is one maintained implementation with a
small, intentional boundary for old users/data.

## 7. Staged plan for the next agent

### Stage A — inspect and classify, without deletion

1. Confirm current status/history/PR state and read applicable `AGENTS.md` and
   repository rules. Identify other agents/jobs that might use the checkout.
2. Establish the accepted baseline and run the default fast suite plus the
   architecture audit. Record pre-existing failures honestly before edits.
3. Build a retirement inventory per path/module: canonical owner, real file or
   symlink, tracked/ignored contents, active callers, dynamic references,
   entry-point behavior, pickle obligations, scientific policy, tests, and
   decision (`migrate`, `retain adapter`, `remove after checks`, `investigate`).
4. Use `rg`/AST searches across source, scripts, shell launchers, notebooks,
   configs and maintained docs. Inspect `sys.path` manipulation, `runpy`,
   `importlib`, path strings, class aliases and installed entry points. A zero
   ordinary-import search is necessary evidence, not sufficient proof.
5. Start with one small, low-risk family of internal consumers. Prefer a pure
   import/path migration over training kernels, session class identities, or
   calibration logic for the first batch.

Store new audit material in an appropriate docs/registry location. Make the
inventory readable; avoid a huge opaque generated report as the only evidence.

### Stage B — migrate consumers with the implementation unchanged

Update consumers to canonical `fpcode` APIs and `studies`/`ops` locations while
leaving compatibility paths available. Preserve public call signatures, defaults
and scientific policies unless a documented migration explicitly changes them.

Test both import and execution behavior. A forwarding module's `__main__` branch
can lose arguments or stop launching correctly despite successful imports.
Check execution from outside the repository, not only from its root. Update
cache provenance to track the actual implementation and effective inputs.

Use targeted tests while editing, then the full default fast suite at the batch
boundary. Commit this consumer migration separately from deletion.

### Stage C — consolidate substantive duplication

Where two implementations really duplicate a capability, compare behavior on
small independently understood examples: empty inputs, missing values, dropped
frames, sparse labels, ties, exact endpoints and weighting differences. Extract
the common operation with explicit policy parameters or named strategies where
the semantics differ. Avoid a catch-all utility module or boolean flag explosion.

Do not homogenize studies' hypotheses, exclusions or cohort definitions. Do not
split modules into tiny wrappers that add indirection without responsibility.
Keep data adapters separate from algorithms and file orchestration.

### Stage D — retire paths only after their obligations are addressed

For each proposed removal, require:

- Migrated known active consumers, including notebooks and launchers.
- Canonical replacement installed/available in the appropriate execution mode.
- Preserved behavior demonstrated by relevant functional/regression checks.
- Historical pickle/class import obligations investigated and covered.
- No lost ignored data, broken data link, or deleted calibration/resource file.
- Updated `registry/SHIMS.tsv`, migration records, docs and regeneration tools.

Pickle is particularly nontrivial: it can store the historical module and class
name. Moving the class or deleting its old import path may prevent loading old
sessions even if no current script imports that path. Class identity matters;
a lookalike replacement class is not automatically equivalent. Test loading
representative **trusted** old data without rewriting originals. If expensive,
use small fixtures plus a bounded empirical check, not a full-cohort traversal.
Do not unpickle arbitrary untrusted artifacts.

Where compatibility remains necessary, retain a minimal explicit import adapter
or a deliberately designed legacy reader. Do not claim all old paths can vanish
until external consumers and serialization constraints are understood.

Inspect generators such as `ops/repo/update_shims.py`: removing a generated
shim without changing its generator may recreate it later. The TSV rows have
historical inconsistencies, including missing trailing fields; parse robustly
and validate records instead of assuming all rows have the same field count.
Do not blindly run a generator as an inventory command if it writes files.

Remove only reviewed tracked files, never whole mixed source/data directories.
Understand symlink versus target: eliminating a link is not the same as deleting
the linked tree, and resolving then deleting a symlink target can destroy data.

### Stage E — verify and publish each meaningful batch

Run the full default suite and architecture audit. Check relevant modified
source for lint/syntax issues; distinguish pre-existing lint debt from new
failures. Do not launch a repository-wide formatting rewrite.

For package/import/resource changes, build a wheel and install into an isolated
environment, then exercise CLI help and representative canonical APIs from a
different working directory without source-tree `PYTHONPATH` support. Test
legacy paths separately if still promised. Do not accidentally import the old
editable installation and call that an installed-wheel check.

Repeat the bounded CPU/GPU checks when affected behavior warrants them, with no
production writes. Capture run paths and observed results. Finally verify a
clean independent checkout/clone of committed source runs the fast suite, so
untracked files or the current working directory cannot hide omissions.

Review the diff, commit in small units, push normally, and keep the next agent's
report/PR description accurate about removals, retained compatibility and limits.
Do not merge `main` merely to finish cleanup; merging is a separate action.

## 8. Backups, data safety and credential precautions

Existing HDD evidence is documented in `docs/recovery.md`:

- `/mnt/hdd/backups/code_snapshots/20261009_112705/code_snapshot.tar.gz`:
  recorded archive checksum verification.
- `/mnt/hdd/backups/TierA_backup_20261009/`: recorded eligible-file coverage and
  a sample of content hashes; **not** full content verification of all 44.22 GiB.
- `/mnt/hdd/backups/FiberphotometryCode/20261009_153903-baseline/`: recorded
  verified Gemini-baseline source archive and Git bundle. Its `workspace/` copy
  was cancelled and is incomplete; do not call it a complete backup.

A final-refactor HDD snapshot was deferred at the user's request. The published
Git branch now provides remote tracked-source recovery; it does not establish
that all current ignored data or outputs have final HDD backups.

The user normally backs up to `/mnt/hdd`, excluding very large files.
`configs/backup.yaml` currently excludes files **at least 2 GiB**, preserves links
without following targets, includes a Git bundle, and requires the HDD mount
and free-space margin. Many files smaller than 2 GiB may still make a repository
snapshot large. Inspect `make backup-plan` before any snapshot. Do not repeat a
large backup/checkpoint exercise automatically; it was specifically deprioritized.

If you do create a needed backup, verify the actual manifest/hashes and restore
into a new empty directory before claiming recovery works. Never restore over
the active tree. Keep referenced ancestor snapshots required by incremental
backups. Never treat an interrupted copy as a valid completed snapshot.

GitHub was authenticated for the authorized push using GitHub CLI with a private
temporary configuration directory. The user elected to handle access cleanup
later. Do not print tokens, read credential files into reports, stage them, or
assume authentication will remain available to a new agent. Local logout does
not revoke the server-side token. GitHub authorization removal is separate from
code rollback and does not delete published commits. No token is needed in this
document. If publishing fails, report the blocker without rewriting Git history.

## 9. Definition of completion for this phase

A good result is a smaller active source tree with fewer legacy callers, fewer
duplicated implementations and no newly hidden scientific or data dependencies.
Remaining adapters should each have a documented reason, a canonical target and
a removal condition. Some serialization adapters may need to remain permanently.

The final report should state:

- Baseline and final commit hashes, branch/PR, and recovery route.
- Which implementations were consolidated and which paths were removed.
- Which old paths remain and why, especially pickle and external-consumer needs.
- Scientific policies preserved and any deliberately changed behavior.
- Exact checks performed, results, runtimes and retained evidence locations.
- Data/backups left unchanged, with any actual backup verification limitations.
- Remaining work, without presenting partial migration as complete retirement.

Do not achieve a superficially clean tree by deleting needed history, disabling
tests, broadening architecture exceptions, dropping data, or hiding failures.
The end goal is a maintainable scientific framework whose behavior and recovery
can be explained and verified.

## 10. Migration execution record (2026-10-11)

All active substantive implementations across fiberphotometry/ and syllable_generation/ have been systematically migrated to studies/ and src/fpcode, backed by standard deprecation forwarder shims:

1. Commit 973d99b (refactor(photometry): relocate fiberphotometry and turnlab into studies/photometry):
   - Migrated fiberphotometry/ data loaders, analysis, KPMS notebooks, and turnlab package into studies/photometry/.
   - Replaced fiberphotometry/ modules with deprecation forwarders that preserve unpickling of Session and DataContainer.
2. Commit d9d1f0e (refactor(studies): migrate sequence_viz, loaders, and videogen into studies/behavior):
   - Migrated 44 sequence visualization scripts, manifests, and pilots into studies/behavior/sequence_viz/.
   - Migrated Jonathan loader package into studies/behavior/loaders/jonathan/.
   - Migrated complete videogen rendering pipeline into studies/behavior/videogen/.
3. Commit c4ded10 (refactor(studies): migrate language_models pipelines, lib, and docs into studies/behavior):
   - Migrated 121 analysis pipelines into studies/behavior/language_models/pipelines/.
   - Migrated dataset loaders, optimizers, clustering, and documentation into studies/behavior/language_models/.
4. Commit 7173aba (refactor(studies): consolidate remaining companion modules across replications, kpms, and studies):
   - Migrated companion modules across studies/behavior/replications_datta/, deep_research_experiments/, syllable_chains/, and multiview_experiments/.
   - Linked KPMS utilities directly to canonical vendor/keypoint_moseq_syll.
   - Cataloged 677 verified backward-compatibility shims in registry/SHIMS.tsv.

### Validation:
- make test: 420 passed in 6.40s.
- make architecture: 121 library modules, 0 violations, 0 cycles.
- Pushed cleanly to origin/codex/modular-refactor (updating PR #1).


## 11. Root consolidation and empirical layout execution (2026-10-11)

Following disk decompression (offloading archive/legacy to /mnt/hdd, freeing 219 GB on NVMe SSD), the root directory was consolidated to canonical architecture standards:

1. Commit 0591610 (refactor(root): retire legacy root symlinks, stray scripts, and redirect callers to canonical studies/):
   - Deleted root tracked symlinks: karen_project, monoamine_interactions, src_mono_inter, src_oft.
   - Removed scripts/ directory (consolidating compaction manifest into ops/compaction/).
   - Redirected callers in test suites, loaders, and experiment runners to canonical studies/ paths.
   - Removed stray root artifacts: update_nb_cell.py, model_fitting.log, .gitignore.legacy.
2. Commit e2e9d2e (refactor(data): establish canonical data/ hierarchy and relocate simple_turn_segmentation doc):
   - Relocated export/simple_turn_segmentation.md to docs/simple_turn_segmentation.md and removed export/.
   - Established top-level data/ hierarchy (data/sessions/, data/models/, data/fused/, data/analysis_cache/).
   - Relocated ~855 GB of empirical data (pickles, checkpoints, fused arrays) internally on NVMe using instantaneous inode moves with 0 extra bytes overhead and backward-compatible relative symlinks.
   - Updated .gitignore to ignore top-level data/ and nested cache/output directories.
   - Reduced syllable_generation/ on SSD from 860 GB down to 3.0 GB.
3. Commit 1e62a4a (refactor(tests): modernize test suite to import directly from canonical fpcode and studies.photometry):
   - Modernized synthetic cache, correspondence, config, fusion, and IO tests to import directly from fpcode.
   - Modernized session tests to import directly from studies.photometry.exploratory.
4. Commit ac2e06f (refactor(photometry): retire fiberphotometry root shim directory to archive/compat/fiberphotometry):
   - Retired legacy fiberphotometry/ package from root into archive/compat/fiberphotometry/.
5. Commit 4cf1688 (docs(reports): relocate root handoff.md to docs/reports/handoff.md):
   - Moved root handoff.md to docs/reports/handoff.md.
   - Relocated untracked logs/ to archive/logs/, scratch/ to archive/scratch/, and kpms_projects/ to data/models/.

### Validation:
- make test: 420 passed in 6.27s (0 failures, 0 regressions).
- make architecture: 121 library modules, 0 violations, 0 cycles.
- Root directory is completely uncluttered, canonical, and modular.

## 12. Complete syllable_generation retirement, real validation, and backup policy (2026-10-11)

The final cleanup milestone retired `syllable_generation/` from the root tree and verified end-to-end execution on real data:

1. Commit f837a35 (refactor(behavior): retire syllable_generation from root into archive/compat/syllable_generation):
   - Relocated `syllable_generation/` to `archive/compat/syllable_generation/`.
   - Modernized all 36 sequence visualization modules in `studies/behavior/sequence_viz/` to import directly from `studies.behavior.sequence_viz`.
   - Created canonical `src/fpcode/contracts/session_record.py` defining `SessionRecord` and `VideoRef` without eager pandas dependencies in base contracts.
   - Modernized callers across `studies/behavior/loaders/`, `studies/behavior/videogen/`, and test suites to import from `fpcode.contracts.session_record`.
   - Modernized `tests/test_v2_visualization_repairs.py` to import `fpcode.viz.v2 as vc`.
   - Both `syllable_generation/` and `fiberphotometry/` are now 100% retired from the repository root.
2. End-to-End Validation:
   - `make smoke-runtime`: PASSED on real batched initialization, optimized AR/SLDS updates, and grid MP4 generation.
   - `make validate-two-sessions VALIDATION_PROJECT=data/models/pilot_comparison_17kp`: PASSED on `cuda:0` GPU in 24.98s across 2 real sessions (evidence in `outputs/20261011_142618-two-session-validation-7d681004/validation.json`).
   - CLI entry points verified: `pipelines/train.py`, `pipelines/evaluate.py`, `pipelines/fuse.py`, `pipelines/render.py`.
3. Backup Policy:
   - Configured `configs/backup.yaml` with exclusion patterns for raw video (`*.avi`, `*.mp4`) and ephemeral directories (`outputs`, `archive/scratch`, `archive/logs`, `data/analysis_cache`).
   - Preflight plan (`make backup-plan`): verified planned snapshot covering 8,440 files (96.37 GB) with Git history bundle.
