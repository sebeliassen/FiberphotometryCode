# Fiberphotometry refactor handoff

Date: 2026-10-09. Working branch: `codex/modular-refactor`.

## Status and scope

The shared-library foundation and initial consumer migration are implemented.
The agreed capability-based tree remains the target; this is not a claim that
every historical script or study has completed migration. No empirical model,
video or recording was relocated. No real training, GPU computation, recording
movie encoding or empirical notebook execution was performed.

**Validation: 232 functional tests passed in 3.49 seconds.** The installed-package
smoke check passed, including all four command entry points and calibration files.
The dependency audit found 115 modules, zero forbidden dependencies and zero
static import cycles. Syntax/undefined-name checks passed.

An implementation commit was already created before the request to stop further
checkpoint work: `6ca24f3`, tagged `codex-refactor-implementation-20261009`.
The Gemini baseline is `0891de1b1c288f16c42d23d7e4b4cbdb020f8cee`, tagged
`pre-codex-modularization-20261009`. This handoff, documentation updates, registry
updates and two whitespace fixes remain uncommitted. No additional final HDD
archive, restore drill or documentation checkpoint was created after that request.

## Implemented structure

```text
src/fpcode/
  contracts/      sessions, timelines, signals, pose, events, artifacts, model
  adapters/       acquisition, dlc, kpms_results, legacy_sessions
  alignment/      clocks, frame_mapping, resampling
  tracking/       geometry, quality, registration, fusion, static_fusion
  photometry/     preprocessing, event_windows, response_metrics,
                  session_signals, aggregation, legacy alignment
  behavior/       kinematics, bouts, transitions, sequences/{tiling,timelines}
  kpms/           preprocessing, training, evaluation, results, checkpoints,
                  backend/{configuration,batching,state,residency,patching,
                           patch_registry,kernels/*}
  statistics/     summaries, resampling
  viz/            selection/{eligibility,density,exemplars,bouts},
                  pose_geometry, grid_movies, trajectories, figures,
                  hashing, provenance, isolated legacy defaults/I/O
  workflows/      fusion, training, evaluation, rendering, photometry,
                  multimodal_analysis
  cli/            train, evaluate, fuse, render
  common/         generic configuration, paths, logging, I/O, atomic publication,
                  cache manifests and run records

tests/            unit/, integration/, regression/, architecture/, golden/
ops/              backup/{snapshot,coverage}, repo/dependencies.py
registry/         checkpoints/20261009-modular-refactor/
docs/             architecture/, decisions/, recovery.md
```

Existing compatibility files coexist with these canonical names. Existing
`environment.lock.yml`, studies, vendor, archive and data locations were retained.
Some older tests remain at the top level of `tests`; default testpaths explicitly
select small functional cases and the new capability-oriented test directories.

## What changed and why

### Shared contracts, adapters and alignment

`TimeAxis`, `SignalSeries`, `PoseSeries`, `EventInterval` and `Bout` encode units,
clock identity, shapes, labels, half-open bounds and validity. Array contracts own
read-only copies of inputs. `SessionKey` and artifact references avoid coupling
new APIs to historical session classes.

Acquisition, DLC and KPMS adapters decode small external tables/files without
loading the model backend or applying scientific corrections. The legacy session
adapter accepts both existing photometry storage conventions and metadata forms.

Strict resampling supports exact, nearest and linear alignment. Nearest alignment
requires a tolerance; linear interpolation requires a maximum gap. Neither fills
outside the source extent. Clock transforms require explicit calibration and
preserve small offsets in integer nanosecond timestamps. Frame mapping places
model labels only on actual original frames; missing labels remain invalid.

### Photometry, behavior and actual consumer reuse

Motion correction, region preparation, aggregation, mean/CI summaries, response
metrics and event windows now have shared implementations. Event windows return
values and validity shaped `(events, offsets, channels)` with explicit clocks.
Photometry and multimodal workflows compose these primitives; a tiny CSV/HDF/DLC
integration test exercises actual decoding, calibration and gap-preserving joins.

Both monoamine `04_signal_peths` and `05_sliding_causality_peth` scripts now call
shared preparation/window helpers. Their paired notebook code cells were updated;
saved outputs and metadata were preserved. Exploratory photometry wrappers and
the monoamine utility module also delegate shared behavior.

KPMS presence rules and fusion smoothing now use named behavior implementations.
Bout extraction and sequence tiling/timeline operations moved into focused
behavior modules. Legacy sampling wrappers retain their historical policies.

### KPMS backend and visualization decomposition

The approximately 1,275-line runtime became configuration, batching, state cache,
residency, patch lifecycle and five kernel modules. The approximately 1,200-line
V2 visualization helper became selection, geometry, hashing, provenance and
legacy-context modules. Old runtime/V2 imports remain facades.

A versioned fixture records baseline function/class AST hashes. **85 relocated
bodies match those saved baseline hashes.** Explicitly changed configuration,
patch lifecycle and hash-publication functions are excluded from that identity
check and have functional checks where applicable. AST equality protects the
relocation; it does not demonstrate real sampler numerical equivalence.

Patch installation is transactional and restores each target namespace's own
entry value, including nested scopes and exceptions. Training configures the
environment before importing JAX, scopes checkpoint/optimization patches and
checks the final checkpoint iteration. Completed or later checkpoints reject a
new run explicitly. Expensive backend calls were faked in small orchestration
tests; real fitting was not executed.

### CLI, configuration, packaging and provenance

Installed commands are `fpcode-train`, `fpcode-evaluate`, `fpcode-fuse` and
`fpcode-render`. Historical `pipelines` imports remain aliases. Nested training
settings reach the fitting implementation; conflicting settings fail explicitly.

Evaluation now uses real parsed arguments and checkpoint compatibility checks.
Rendering honors requested artifact types, syllable subsets and exact tile
counts, and fails on missing inputs. Fusion forwards supported options and
reports cached/failed/empty work correctly.

Run records hash effective execution settings separately from source config
path/hash, copy caller settings and use shared atomic JSON publication. Base
package/CLI/runtime configuration imports avoid JAX, Torch and CUDA startup.
Installed packages honor explicit roots and otherwise use the working directory.

Cache fingerprints reference actual moved implementations. Removing the last
input is detected; malformed JSON shapes are rejected; missing implementation
files cannot silently produce a stable fingerprint. Legacy visualization hashes
also include extracted implementation files, preventing shim-only cache identity.

The wheel contains canonical `fpcode` packages and all three tracking calibration
JSON files. An offline temporary build was checked from outside the checkout;
four metadata-defined entry points successfully displayed help. This used the
existing project environment, not a fresh dependency installation.

## Scientific compatibility to preserve

- Historical photometry interpolation is by row position, with its existing
  endpoint/gap filling. Strict coordinate interpolation is a separate API.
- Dense timeline filling and sparse timelines with gaps remain distinct.
- KPMS and behavioral presence rules retain their different logic; smoothing
  variants retain short-array/window behavior.
- Existing tuple metric interfaces remain compatible. Historical first-sample
  slopes retain their undefined masked-division behavior; correcting that requires
  an explicit scientific decision and new independent expectations.
- Raw tracking confidence, calibration and scientific cohort choices remain
  explicit policies. Do not unify algorithms solely because their names match.
- Old imports and class paths remain available. No empirical pickle was opened
  to certify reconstruction, so keep those paths until that check is complete.

## Verification and how to resume

```bash
/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python -B -m pytest \
  -q -p no:cacheprovider --basetemp=/tmp/fpcode_next_checks

/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python -B -m ruff \
  check src/fpcode --select E9,F63,F7,F82

make architecture PYTHON=/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python
```

The final suite passed 232 tests with eight expected legacy deprecation warnings.
Coverage includes independent values/masks, immutable inputs, strict/legacy
alignment distinctions, real tiny CSV/HDF/PNG outputs, parsed CLI arguments,
training endpoint/cleanup behavior, cache invalidation, backup corruption and
fresh-directory restoration using tiny fixtures. CPU-only is not a runtime
guarantee; large empirical tests are outside the default testpaths.

Versioned evidence is in `registry/checkpoints/20261009-modular-refactor/`:
`validation.json`, `dependencies.json` and `packaging.json`. Disposable packaging
details are also in `outputs/refactor-20261009/`. The baseline extraction fixture
is `tests/golden/module_extraction.json`. The dependency audit checks absolute and
relative imports, static cycles, syntax failures and import-time argument parsing;
computed dynamic imports still require human review.

## Existing backups and rollback

The other agent's backups were reused:

- `/mnt/hdd/backups/code_snapshots/20261009_112705/code_snapshot.tar.gz`:
  existing archive checksum verified.
- `/mnt/hdd/backups/TierA_backup_20261009/`: all 509 expected eligible artifacts
  present, totaling 47,480,759,461 bytes. Path/type/size coverage and ten bounded
  full-SHA256 spot checks passed. Four additional files are recognized provenance
  metadata. **The complete 44.22 GiB mirror was not rehashed.**
- `/mnt/hdd/backups/FiberphotometryCode/20261009_153903-baseline/`:
  verified archive of finished Gemini code and verified Git bundle, with checksums
  in `code_baseline.json`. Its redundant `workspace/` copy was cancelled and is
  incomplete; do not treat `manifest.json` as a successful backup.

No new full data backup was made. New snapshot tooling provides plan/create/
verify/restore, excludes files at least 2 GiB, records exclusions, preserves
symlinks without following targets, verifies full contents and restores only to
fresh destinations. Incremental references pin ancestor manifests and contents;
keep ancestors. The legacy full verifier remains expensive by design.

For a safe source comparison use the baseline tag. To inspect earlier code
without replacing the workspace, create a separate worktree:

```bash
git worktree add /tmp/fpcode-before-refactor pre-codex-modularization-20261009
```

See `docs/recovery.md`. Git rollback covers tracked code, not empirical data.
Do not use broad reset/clean commands while someone is working in this folder.

## Remaining work, in priority order

1. Review the implementation diff and this handoff against the agreed tree.
   Documentation/registry changes are deliberately uncommitted after the user's
   request to prioritize the handoff over further checkpoints.
2. Validate representative historical session pickle imports and one small
   empirical photometry/behavior comparison before retiring compatibility paths.
3. Run separately authorized backend checks for relocated KPMS kernels and one
   real render/fitting smoke test. Functional fakes and AST checks are not these
   gates; keep large benchmarks out of the default suite.
4. Continue migrating repetitive preparation in other monoamine notebooks and
   cohort runners. The old cohort exporter in `kpms/export.py` still mixes study
   orchestration and legacy session mutation; migrate it into explicit workflows
   only after characterizing its scientific conventions.
5. Wire new tracking geometry/quality/registration and artifact contracts into
   additional production consumers where their behavior is equivalent. They are
   foundation APIs, not evidence that every legacy implementation was replaced.
6. Finish test organization and remove duplicate V2 unittest/function collection.
   Extend independent regression fixtures for response edge cases and sequence
   boundaries before changing historical scientific policies.
7. Add study-level configuration/adapters where actual consumers need them.
   Retain isolated V2 cohort defaults for compatibility; avoid empty scaffolding.
8. When useful, create a final code-only HDD recovery snapshot and restore drill.
   This was deferred explicitly; no final-refactor archive should be assumed to
   exist. Current post-baseline code is in Git and the working tree.

The architecture decision is documented in
`docs/decisions/0001-capability-modules-and-compatible-scientific-policies.md`.
Recursive modularity means coherent capability boundaries, not a mandatory CLI,
YAML file and README in every leaf folder.
