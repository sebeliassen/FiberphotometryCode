# Antigravity continuation handoff — 2026-10-11

This is the authoritative handoff for the interrupted Claude-review follow-up and project-first structural refactor. It is written for an agent without the conversation history. **The work is incomplete. Read this document before changing files, regenerating shims, or running migration utilities.**

The user approved addressing both the library concerns and the structural cleanup, then explicitly asked Codex to stop implementation and produce this handoff. Implementation has stopped; no subagent has a pending mutation or running command. Creating this document and preserving its supporting evidence were the final actions. Continue implementation in the next session when the user gives you this handoff.

## 1. Checkout, environment and immediate state

| Item | Value |
| --- | --- |
| Workspace | `/home/fsp585/Desktop/FiberphotometryCode` |
| Branch | `codex/modular-refactor` |
| HEAD before and after this session | `409407a06fb29e9658b6d75c958703dd6ab22978` |
| Working tree | Extensive uncommitted changes and physical moves; initially clean |
| Validated interpreter | `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python` |
| Commits/pushes/merges in this session | None |
| Final default functional suite | **436 passed, 4 failed, 7 warnings in 14.22 seconds** |
| Final architecture audit | **140 modules, 0 violations, 0 cycles** |
| Lint | Incomplete; strict-source errors and missing reviewed debt inventory |
| Mypy | Not installed in the existing environment; not run |

Use the existing interpreter for local checks. System Python lacks the necessary tooling. Do not upgrade or reinstall the shared lab environment to make a check pass. A separate temporary environment is appropriate for reproducibility testing later.

Before these changes, the default suite passed 420 tests with one warning in 6.38 seconds, and the audit covered 122 modules with no violations/cycles. Discovery now includes additional synthetic checks and new ownership tests; do not compare pass counts as if both runs collected the same suite.

Paths in this document are relative to the workspace unless explicitly absolute. The historical root `antigravity_refactor_handoff.md` and earlier reports describe an earlier checkpoint. They remain useful background, but this document supersedes their continuation status.

Immediately before the handoff files were created, Git porcelain contained **73 modified, 2,173 deleted and 86 untracked entries**. These are status entries, not a count of permanently deleted files. Many tracked old paths were physically moved to untracked new directories. An unstaged `git diff` does not show those new destination files. Untracked directory entries also aggregate many files.

**Do not run `git clean`, `git reset --hard`, broad `rm -rf`, or `git add -A`.** Reconcile moves with the ledgers and stage source deliberately. Several source destinations are ignored by broad existing ignore rules; fixing files locally does not guarantee that Git will include them.

The Codex shell sandbox failed even harmless commands with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`. Read-only checks and bounded authorized writes worked through reviewed escalation. A proposed broad library rewrite was rejected before execution; accepted work used scoped patches. There is no unresolved approval blocker. This is tooling context, not a reason to stop ordinary authorized work.

## 2. Evidence to read first

The source-only support directory is [handoff_20261011](handoff_20261011/README.md). Its evidence has been copied out of `/tmp`; **do not depend on temporary files surviving**.

| Evidence | Purpose |
| --- | --- |
| `registry/STRUCTURE_MOVES.tsv` | 37 physical study, compatibility, policy and data/output moves |
| `registry/TEST_MOVES.tsv` | 38 test/notebook/manual/example source moves |
| `registry/LINK_REPAIRS.tsv` | 81 repaired symlinks and target identity verification |
| `handoff_20261011/claude_review.txt` | Original external review, preserved verbatim |
| `handoff_20261011/checkpoint_state.json` | Pre-document Git counts and exact three dangling links |
| `handoff_20261011/architecture.json` | Final read-only audit, including dependency graph |
| `handoff_20261011/reference_counts.json` | Residual-reference scan; indicators, not proven failures |
| `handoff_20261011/symlinks_before_cleanup.json` | Earlier symlink inventory: 837 links, 84 dangling |
| `handoff_20261011/link_repairs_reviewed.json` | Explicit approved subset of repair decisions |
| `handoff_20261011/fpcode_migrate_paths.py` | Unfinished scoped path/import rewrite helper; inspect before using |
| `handoff_20261011/path_migration_originals.json` | Exact original text and SHA-256 for 77 files before scoped path rewrites |
| `handoff_20261011/prior_lint.json` | 3,254 diagnostics in 709 maintained Python files from the unchanged Git HEAD |
| `handoff_20261011/fpcode_prior_lint.py` | Historical lint-snapshot extraction helper; not a task to rerun blindly |
| `handoff_20261011/SHA256_MANIFEST.json` | Sizes and hashes of the preserved evidence files |

The 77 source originals are **post-move, pre-path-rewrite** text, not the Git HEAD baseline. The baseline commit is the authority for earlier source. Preserve the original numerical golden fixture at `tests/golden/module_extraction.json` independently of both.

Other useful documents: `docs/architecture/UNIT_CONTRACT.md`, `docs/recovery.md`, `docs/reports/refactor_report.md`, `docs/reports/cleanup_20261011.md`, and `tests/README.md`. Check actual paths before relying on historical links. `cleanup_20261011.md` describes some intended outcomes and still says integration verification is pending; it must eventually reflect the actual final state.

## 3. Accepted design and qualifications to Claude's review

The destination is a project-first workspace:

```text
src/fpcode/       reusable algorithms, contracts, adapters, workflows and CLI
studies/         project-owned policies, analyses, notebooks and launchers
ops/             operational implementations grouped by purpose
compat/          live historical filesystem/import adapters
data/            raw recordings, model artifacts, fused inputs and sessions
outputs/         generated run products and retained historical outputs
configs/         environment/configuration definitions
tests/           synthetic unit, integration, regression and architecture checks
                 plus explicit empirical opt-in
docs/            maintained guidance and clearly dated historical reports
registry/        migration, compatibility and artifact inventories
vendor/          vendored dependencies
archive/         inactive historical material and eventually frozen bundles
```

This is a responsibility map, **not a claim that the root now has exactly twelve entries**. Root `core/` and `pipelines/` still contain physical historical adapters. Temporary aliases under `studies/behavior`, `studies/photometry` and `archive/compat` also remain. The root `reports/` bundles have not been physically archived yet.

The user accepted these qualifications in our discussion:

- Repair filesystem references as well as Python imports. Claude's “about 390 broken files” was a suspicion from a ZIP lacking symlink metadata, not a verified failure count.
- Keep necessary physical shims. A global `sys.meta_path` finder cannot provide a file for direct script execution, `runpy.run_path`, explicit file/spec loading, notebook paths or filesystem callers. It also adds identity and startup complexity. A reviewed mapping/registry can be the source of truth without replacing all physical adapters with an import hook.
- Distribution name `fiberphotometry-suite` and import name `fpcode` are legitimately different. A rename is unnecessary. The duplicated `all`/`kpms` dependency extra can still be cleaned up separately.
- Small forwarding modules are not duplicated numerical algorithms. Move substantive historical policy to `fpcode.compat`, preserve public paths and shared object identity, and enforce explicit boundaries.
- Do not rename every historical report or rewrite scientific history merely to remove old paths. Correct maintained guidance; mark dated reports clearly.
- The root `REFACTOR_REPORT.md` duplication was generated in the earlier ZIP, not an independently maintained duplicate in the current checkout.
- Scientific ownership matters more than minimizing the number of directories. Cohort IDs, calibration selections, chamber/LED choices, sampling policy and exclusions stay explicit and study-owned or behind a named compatibility profile.

### Coverage of the original review

| Concern | What landed | What remains |
| --- | --- | --- |
| Retired `syllable_generation` paths | Ledgers, asset moves, 81 link repairs, three scoped source rewrites | Most study consumers, notebooks and constructed paths; shim targets |
| Inverted ops relocation | 10 implementations moved to purpose directories; `ops/scripts` aliases | Registry integration and ignored diagnostic source staging |
| Live code under `archive/compat` | Physical move to root `compat` | Temporary `archive/compat` alias and residual consumers |
| Data under studies | Known large raw/session/output directories moved with inode checks | Transitional aliases, other caches and remaining test-root binaries |
| Loose tests/notebooks | 38 source moves; full `tests` discovery; empirical opt-in | Four failing tests, minimal CPU dependencies, manual launcher repairs |
| Stale docs/reports | Maintained ops/recovery/README updates and historical banners | Final factual status, remaining links, frozen bundles |
| Compatibility ownership | Substantive historical implementations in `fpcode.compat` | Some older wrappers, deeper responsibility cleanup and policy injection |
| Dependency matrix | Two unused edges and export exception removed; tracking primitives lowered | Future extracted modules must respect the same matrix |
| Common I/O domain logic | Generic file I/O separated; backend-free KPMS checkpoint I/O added | Strict lint and historical CLI alias integration |
| KPMS metrics/statistics | Existing interfaces retained | Backend parsing versus pure counting/statistics split not implemented |
| Calibration ownership | Three unchanged JSON resources moved to named compatibility profile | Study profile injection can be improved; no generic-domain defaults added |
| Shims | Existing physical execution/identity behavior preserved | Canonical targets and `SHIMS.tsv` still need regeneration/review |
| Typing/CI/packaging | `py.typed`, wheel check, optional typing and CPU workflow added | Lint inventory, mypy, actual clean CPU/remote CI verification |

## 4. Completed physical moves and their limits

`STRUCTURE_MOVES.tsv` records the authoritative old/new locations. Same-filesystem moves preserved source/data inodes; large empirical payloads were not copied, modified, unpickled or rehashed. When destination source directories already existed from concurrent notebook organization, nonconflicting entries were merged with inode checks.

Study source now lives directly under:

```text
studies/kinematics_ulrik
studies/language_models
studies/sequence_viz
studies/multiview_experiments
studies/deep_research_experiments
studies/shmoseq
studies/replications_datta
studies/karen_pff_vs_monomer/analysis
studies/karen_pff_vs_monomer/session_policy
studies/photometry_exploration
studies/turnlab
studies/oft_baseline
studies/oft_baseline/ifdn_policy
studies/juvenile_intruder
studies/session_building
studies/multimodal_movies
studies/sequence_viz/chain_analysis
studies/syllable_annotation
studies/session_model_exports
studies/model_inspection
```

`studies/monoamine_interactions` remains a source owner, but its old project policies, sessions, logs and outputs have separate destinations. The OFT policy directory was merged into `studies/oft_baseline` after its raw assets moved out; raw aliases remain there temporarily.

Important empirical destinations:

- Karen videos, bottom videos and HDF5: `data/raw/karen_pff_vs_monomer/{video_files,botcam_video_files,h5_files}`.
- OFT recordings/workspaces: `data/raw/oft_baseline/`; approximately 280.6 GB moved by rename, not a new backup.
- Old `studies/mono_inter_juvenile_intruder`: `data/raw/juvenile_intruder/`. This includes juvenile **and baseline** subdirectories; do not infer its contents from the new parent name.
- Monoamine sessions: `data/sessions/monoamine_interactions/`, approximately 1.13 GB, same inodes.
- Monoamine historical outputs/logs: `outputs/monoamine_interactions/legacy/` and its `logs/` child.
- Ulrik historical output: `outputs/kinematics_ulrik/legacy/`.

Temporary old-location symlinks are intentional until consumers and compatibility obligations are repaired. Their presence does not mean data is still physically stored in the source directories. Conversely, retaining them is not proof that all constructed paths work.

### Three current dangling links

The exact final list is in `checkpoint_state.json`:

1. `compat/syllable_generation/context/tandem_viz_guide.md` points to the removed absolute `.../syllable_generation/multiview_top_bot_experiments/tandem_viz_guide.md`. This guide target was already missing. Investigate retained documentation; do not fabricate a replacement.
2. `studies/session_model_exports/name_syllables` points to `../../../../syllable_annotation`.
3. `studies/session_model_exports/syllable_chains` points to `../../../../sequence_viz/chain_analysis`.

The latter two are **new transition mistakes**: those relative child aliases moved with their parent and now have the wrong depth. Verify the actual sibling destinations, then repair toward `../syllable_annotation` and `../sequence_viz/chain_analysis`. They are not all pre-existing problems.

The 81 completed link repairs used explicit reviewed mappings, temporary links and atomic replacement; destination device/inode/size checks confirmed target identity. Valid external links under `/mnt/hdd` and external Desktop dataset roots were deliberately left unchanged. Do not traverse directory symlinks during bulk cleanup or remove external targets.

## 5. Completed library and ops work

### Library ownership

New substantive compatibility owners include:

```text
src/fpcode/compat/io.py
src/fpcode/compat/schemas.py
src/fpcode/compat/kpms_export.py
src/fpcode/compat/multiview/profile.py
src/fpcode/compat/multiview/resources/*.json
src/fpcode/compat/photometry/frame_merge.py
src/fpcode/compat/photometry/acquisition_alignment.py
src/fpcode/compat/viz/{v2,legacy_api,legacy_defaults,legacy_io}.py
```

Old public paths are thin forwarding facades. `ops/repo/dependencies.py` names the ten permitted facade targets explicitly; it rejects substantive definitions in those facades and direct domain imports of compatibility owners. Workflows can select an explicit compatibility profile. Existing historical tracking APIs still reach defaults indirectly through `adapters.multiview_compatibility`; **the current audit is a direct-boundary check, not universal transitive isolation**.

- Generic HDF5/NumPy/pickle file utilities moved to `common/file_io.py`.
- Backend-free checkpoint inspection/compaction moved to `kpms/checkpoint_io.py`; original snapshot group names and nonnumeric groups are preserved, retention is validated, and publication uses a temporary file and atomic replacement. No real checkpoint was compacted.
- Circular/vector median primitives moved to `statistics/smoothing.py`. Tracking no longer imports behavior for those kernels. Historical wrapper signatures and behavior policy branches remain.
- `contracts/session_record.py` no longer eagerly imports pandas. Class identities and qualified names remain unchanged. Runtime `typing.get_type_hints` may need a supplied pandas annotation namespace; do not restore the eager import just to resolve annotations.
- `PathConfig` now uses repository/data resolvers and `FP_BOT_TRACKING_ROOT` instead of hardcoded workstation defaults. `PathConfig.__module__` intentionally remains `fpcode.common.schemas` for old pickle names.
- Three calibration JSON files were verified byte-for-byte before moving from tracking to compatibility resources. Their scientific content, including historical timestamp strings, was preserved.
- Removed unused allowed edges `viz -> kpms` and `kpms -> photometry`, and the `kpms.export -> workflows.session_export` exception. Forbidden library roots include studies, historical roots, tests, scratch, core and pipelines.
- Added `tests/architecture/test_compat_boundaries.py` and `tests/unit/test_library_ownership.py`.

`fpcode.viz.v2` deliberately remains a distinct physical facade with `__name__ == "fpcode.viz.v2"`; its exported callables share identity with compatibility implementations. Do not replace it with an unconditional module alias. Other legacy facades intentionally share a module namespace to preserve monkeypatch behavior.

Fusion provenance still needs repair: both tracking `_FUSION_CODE_FILES` lists fingerprint the old `adapters/multiview_compatibility.py` and `common/schemas.py` facades but omit their actual `compat/multiview/profile.py` and `compat/schemas.py` implementations. Include the implementation dependencies and verify cache invalidation when those implementations change. The completed V2 fingerprint expansion does not fix fusion caches.

Resolved defaults are not verified existing inputs. `PathConfig` currently defaults to nonexistent `data/raw/karen_pff_vs_monomer/bot_tracking_predictions` and `data/models/kpms_tandem_cohort_production`. Callers must configure the real data/model locations, including `FP_BOT_TRACKING_ROOT` for external predictions. Preserve existing external data; do not create empty directories or replacement artifacts to make an existence check pass.

**Still not implemented:** the KPMS metrics parsing/counting split, deeper `behavior.kinematics` responsibility cleanup and remaining migrated docstrings. Tiny aliases such as `photometry.metrics`, `statistics.summary`, `viz.geometry`, `kpms.data` and `kpms.fit` remain and do not themselves duplicate algorithms.

### Export CLI integration remains unfinished

`fpcode.cli.export:main` and installed `fpcode-export` now perform checkpoint compaction, with positive retention validation and a dry-run that writes nothing. They pass the actual list of retained iterations and an explicit destination correctly. Default in-place compaction is historical behavior and is described in help.

**`pipelines/export.py` is still the old 127-line implementation.** It passes an integer as `keep_iterations` and inaccurately advertises results extraction. Convert it to the same physical/module-forwarding pattern as the other historical pipeline verbs, preserving script execution and public imports. Do not implement or claim results export merely because the old help says it exists. Test using a tiny synthetic checkpoint; do not run against production data.

### Ops implementation direction was repaired

Ten former `ops/scripts` implementations now belong to:

- `ops/monitoring`: `ntfy_monitor`, `ram_watchdog`, `launch_extended_when_limited_done`, `notify_kpms_iterations`.
- `ops/dlc`: `verify_botcam_orientation`, `relink_split_videos`.
- `ops/compaction`: `batch_compact_all_checkpoints`.
- `ops/repo`: `build_swe_archive`, `execute_phase3_moves`, `update_nb_cell`.

`ops/scripts` now provides old import aliases and direct execution through `runpy.run_path`. The compaction script exposes `run_pipeline`, not `main`; preserving execution cannot depend on every target having `main()`.

Karen-specific link/rsync operations point at the study-owned analysis implementations. Missing historical launch inputs are now explicit arguments where possible, and subprocess launchers use the current interpreter. Once-off historical migration/triage tools are labeled historical. **Do not rerun them as a current-tree inventory.**

The ops agent checked syntax/undefined names, parsed all ops Python, verified ten alias module identities, eleven harmless help commands, and three shell scripts with `bash -n`. It did not launch monitoring, DLC, real compaction, relinking, rendering, notebook patching or notifications.

## 6. First repairs: the four failing regressions

Run these targeted tests first after the initial filesystem inventory:

```bash
cd /home/fsp585/Desktop/FiberphotometryCode
export PYTHONDONTWRITEBYTECODE=1
FPCODE_PY=/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python
"$FPCODE_PY" -B -m pytest -q -p no:cacheprovider \
  tests/architecture/test_study_behavior_imports.py \
  tests/regression/test_module_extraction.py
```

These four failures are all failures in the final default run; the rest of that run passed. Do not weaken these assertions to obtain a green result.

### Three historical/canonical identity failures

The failing tests are:

- `test_study_behavior_imports_work_in_fresh_interpreter_without_eager_config`.
- `test_historical_metrics_split_exports_keep_identity_without_import_cycles[False]`.
- The same test with `[True]`.

Tests now import canonical `studies.kinematics_ulrik` and `studies.replications_datta`, but physical compatibility shims still target `studies.behavior.kinematics_ulrik` and `studies.behavior.replications_datta`. Directory symlinks let Python execute identical source under two distinct qualified module names. Same inode/source text does **not** imply the same module, function or class object.

Repair compatibility targets and relevant internal study imports to the canonical names. Preserve `sys.modules` sharing where required. The tests intentionally cover both import orders, historical bare `analysis...` names, lazy split exports, exact `__all__`, unknown-attribute behavior, old function `__module__` obligations and absence of eager scientific backend/rendering imports.

Do not delete temporary directory aliases first. Do not create another copy of the source. Do not replace identity assertions with equality, remove subprocess isolation, import heavy config eagerly or change historical pickle namespaces.

`registry/SHIMS.tsv` still contains the earlier 677 ACTIVE rows and has not been regenerated for these moves. `ops/repo/update_shims.py` is a historical subset updater with stale paths; running it now can write stale mappings or reconstruct the old namespace. Review the registry schema and missing/variable fields before writing a manifest-based replacement. Reconcile filesystem targets and import targets explicitly; do not assume every row is a current executable module.

### Golden AST lookup failure

`test_extracted_numerical_bodies_match_pre_refactor_baseline` raises `KeyError: 'get_session_files'`. It looks for definitions directly in the file named by the saved fixture, but `viz.legacy_io` is now a thin facade and its real definitions are under `compat.viz.legacy_io`.

Add an explicit current-location mapping for relocated implementations in the test. Parse the actual implementation and compare the **same saved AST digest**. Preserve the original golden fixture and the invariant that 73 non-excluded bodies are checked. Determine all affected relocated modules rather than patching only the first missing name.

Do not regenerate `module_extraction.json` from current code, skip missing definitions, broadly expand `INTENTIONAL_CHANGES`, or reduce the count. If a body actually changed, examine the change and provide independent numerical expectations before making a narrow, justified test adjustment.

## 7. Source path migration is only partially applied

The preserved `fpcode_migrate_paths.py` was applied **only** to these source owners:

| Owner | Python files changed by the scoped helper |
| --- | --- |
| `studies/session_building` | 1 |
| `studies/multimodal_movies` | 3 |
| `studies/multiview_experiments` | 73 |

Those edits parsed successfully; they are not proof that every script runs correctly. In particular the builder's dynamic project resolution remains semantically unfinished. Other studies were physically moved but did not receive this helper's canonical source migration. Compatibility shim targets were not rewritten by it.

Dry-run estimates for remaining owners were: deep research 15 files, Karen 24, kinematics 27, language models 134, model inspection 1, monoamine 17, photometry exploration 18, Datta replications 5, sequence visualization 57, session model exports 10, SHMoSeq 31, annotation 2 and turnlab 1. OFT and juvenile policy previews proposed zero changes; that does not certify their dynamic behavior.

The final textual scan found:

| Area | Python files scanned | `/home` mentions | Old modality study import indicator | Old `syllable_generation/` indicator |
| --- | ---: | ---: | ---: | ---: |
| `compat` | 637 | 0 | 581 | 572 |
| `studies` | 651 | 264 | 98 | 189 |
| `src/fpcode` | 140 | 0 | 0 | 0 |
| `ops` | 55 | 1 | 0 | 5 |
| `tests` | 55 | 2 | 0 | 3 |

These counts include comments, warnings and historical documentation strings. They count files matching indicators, not individual broken paths or verified launch failures. Classify actual input paths, intentional future outputs, external instrument paths and historical descriptions before editing. Audit notebooks, shell scripts and constructed strings separately.

### Root semantics that must stay explicit

`get_repo_root()` honors `FIBERPHOTOMETRY_ROOT`, otherwise discovers checkout anchors; an installed wheel without anchors falls back to the current directory. `resolve_data_path()` honors `FP_DATA_ROOT`, defaulting to the checkout root.

**`FP_DATA_ROOT` is a mirror root containing `data/`, not the `data/` directory itself.** Existing callers pass paths such as `data/models/...` and `data/raw/...`. Setting it to `/external/data` while retaining those arguments creates `/external/data/data/...`. The strengthened bootstrap test deliberately separates source/config and data roots. Existing module-level path constants also bind at import; use fresh processes when testing environment overrides.

Current V2 path expectations are:

```text
EXP_DIR      = get_repo_root()/studies/multiview_experiments
SCRIPT_DIR   = actual Path(__file__).parent for the source study module
COHORT_DIR   = resolve_data_path(data/models/cohort_static_21kp_v2)
MODEL_DIR    = COHORT_DIR/model_21kp_static_50ar_300slds_seed101
FUSED_DIR    = resolve_data_path(data/fused/multiview/cohort_fused_static_21kp_v2)
VIZ_DIR      = resolve_data_path(data/analysis_cache/multiview_visualizations/
                               visualizations_cohort_static_21kp_v2_seed101_v4)
RAW_INPUTS   = resolve_data_path(data/raw/karen_pff_vs_monomer/...)
```

Config, results, video-frame indexes and preprocessed data belong under the configured model/data roots. `SCRIPT_DIR` records actual source provenance and must not be replaced by an empty synthetic experiment root. The final full suite included the strengthened V2 bootstrap test and it passed.

### Helper hazards: review before applying

The helper is an unfinished aid, not a universal migration engine. Read its source and preview **one real, nonsymlink project directory** at a time:

```bash
"$FPCODE_PY" -B docs/reports/handoff_20261011/fpcode_migrate_paths.py \
  studies/replications_datta
# No --apply above: prints the proposed changed-file count only.
```

It does not print a detailed diff, so inspect proposed `migrate(text, path)` results or add a reviewed preview mode before applying further projects. Its workspace root is hardcoded and its live backup destination is still `/tmp/fpcode-path-migration-originals`. The preserved JSON originals will not automatically update after new writes; choose a durable separate backup destination before proceeding. Do not overwrite the handoff evidence.

Nontrivial cases already encountered:

1. A former experiment directory mixed source, models, videos and cache. Rewriting only `EXP_DIR` would send model lookups into the new source tree. Crossings such as `EXP_DIR / cohort_name` need explicit `data/models` destinations.
2. Dynamic project paths cannot be fixed by replacing literals alone. See the project-policy mapping in the next section.
3. Moving a file changes `Path(__file__).parents[n]`. Verify the intended root; do not preserve the old numeric depth blindly.
4. A trailing slash can be required by old string concatenation. Preserve it while replacing a root expression.
5. Python AST column offsets are UTF-8 byte offsets, not character indexes. The helper converts them before slicing source; keep this behavior for files containing non-ASCII text.
6. Bare fragments such as `"monoamine_interactions"` inside `Path(root)/"studies"/...` are not complete root-relative paths. A global replacement produced a duplicated `studies/studies/...` preview and was corrected before applying it. Do not reintroduce this by text replacement.
7. A syntactically valid rewrite can still point at the wrong policy/data owner. Validate path construction without loading empirical inputs, then smoke-test harmless imports/help with code and data roots separated.
8. Absolute external instrument/data locations require explicit configuration or profile inputs. Do not mechanically relocate them just because they contain `/home`.

`src/fpcode/common/workspace.py` was discussed but **was not created**. Do not reference it as an existing helper.

## 8. Shared session-building and movie extraction: planned, not started

No new extraction implementation or tests landed. Proposed `adapters/session_sources`, `workflows/session_building.py` and `viz/composite_movies` do not exist. This is the largest remaining structural responsibility change, not a completed feature.

### Session building

Inspect `studies/session_building/build_sessions_unified.py` and its Jonathan builder/config/context/steps/utilities. These currently use bare `config`, `enrich` and `tracking` imports, `sys.path` mutation, module reloads and global `JCFG` changes. That makes policy ownership and repeated execution fragile.

Resolve known projects explicitly to their owning policy modules/directories:

| Existing project choice | Owner |
| --- | --- |
| `jonathan_full`, `jonathan_openfield_example`, `jonathan_openfield_amph` | `studies.oft_baseline.ifdn_policy` |
| `jonathan_oft_baseline` | `studies.oft_baseline` |
| `jonathan_juvenile_intruder` | `studies.juvenile_intruder` |
| `karen_project` | `studies.karen_pff_vs_monomer.session_policy` |

**Do not append the project name again after selecting its complete owner path.** The partly rewritten dynamic builder still needs this correction. Preserve old accepted choice names through a resolver rather than renaming user inputs silently.

`BuildContext` default factories and step modules imported `JCFG` directly. Reassigning `config.JCFG` does not update names already bound in other modules. Canonical mechanisms should receive explicit `ctx.cfg` and callbacks; compatibility study wrappers should preserve the old constructor order/defaults. Test two policies in one process to prove there is no state leakage.

Suggested ownership:

- Source-format parsing and configurable discovery: adapters.
- Scientific photometry processing: photometry.
- Generic pipeline orchestration: workflows.
- Chamber LED/channel maps, sampling rates, exclusions, cohort file maps and selected recipes: studies/configured policy.

The library must never import `studies` to discover a policy. `studies/monoamine_interactions/steps/tracking.py` has a similar bare-config discovery mechanism; extract generic discovery with explicit configuration and keep cohort overrides as callbacks. Preserve the canonical `SessionRecord` class instead of defining a second class with the same fields.

Numerical invariants requiring independent expectations:

- Nearest-timestamp ties choose the left neighbor; reference order remains stable.
- Missing frame mappings remain `-1`.
- Preserve all acquired LED channels and their schema.
- Interpolate reference signals before the selected correction step.
- Preserve rolling windows and distinct Wallace/manual/RANSAC recipes. Historical RANSAC 20 Hz/3 Hz values are policy, not an opportunity for an untested “cleanup”. Expose parameters before changing them.

Do not overwrite existing `photometry/session_signals.py`: its `PreparedSignals` and `prepare_region_traces` already have their own responsibility. Possible new acquisition-metric/session-processing modules were suggestions, not written files.

### Movie composition

Inspect `studies/multimodal_movies` composer, grids, utilities and verification scripts. Generic video/photometry/syllable-grid composition can receive explicit dimensions, style, FPS, PDF inputs and offsets. Study IDs, mouse selections, colors/rates and source/output path policy stay with the project.

Preserve these differences and edge cases:

- Duo/trio composition uses `source_fps / target_fps`, floor frame counts and `start + int(i * stride)` source positions.
- `utils.resize_and_pad` explicitly selects OpenCV AREA/LINEAR interpolation; `grids.video` uses OpenCV's default. They are not interchangeable implementations even if most images look similar.
- A missing syllable PDF keeps the last valid image in the existing grid behavior.
- Avoid setting a Matplotlib backend at module import. Load rendering resources lazily.
- Release capture/writer/figure resources on exceptions and early returns without changing emitted frame selection.
- Verification scripts' 40 FPS/every-second-row sampling is explicit scientific policy.

Inspection found a probable verification-reader off-by-one and a photometry fallback-key issue (`sig[1]` versus `sig[0]`). Treat them as separate behavior repairs with tests; do not silently change them inside a claimed equivalent extraction.

Proposed `test_session_source_extraction.py` and `test_composite_movie_extraction.py` were **not created**. Useful tests should use independent expected timestamps/signals and tiny or fake media, cover configuration isolation and resource cleanup, and avoid merely asserting the new implementation calls itself.

## 9. Test, lint, packaging and CI integration

### Test organization and empirical guard

`TEST_MOVES.tsv` records 18 moved test modules, 11 notebooks, 6 manual scripts and 3 visualization examples. Five data/backend modules now live under `tests/empirical`. `pyproject.toml` uses `testpaths = ["tests"]` and source paths `.`/`src`/`compat`/the vendored KPMS utility directory. Logs, PID files and large `pkltest.pkl` / `salient_kappa_3e10.pkl` were still found under tests; they were not unpickled or moved by the test agent.

`tests/conftest.py` adds `--empirical`. Normal directory discovery skips empirical modules **before import**. An explicit-path guard was also added because pytest can bypass ignore-collection hooks for a named file. `--empirical` only permits collection; default `not gpu and not slow and not needs_data` marker selection still prevents those bodies running. Select runtime markers explicitly for an authorized empirical run.

Empirical imports/loaders were made lazy. Earlier blocked-backend collection found 16 tests in 0.16 seconds without importing JAX/JAX-MoSeq/Keypoint-MoSeq. That check preceded the final explicit-path guard, so verify the guard independently and repeat backend-blocked collection after finishing changes. Collection is not empirical execution.

Six configuration comparisons now read literal training-runner constants via AST instead of executing a production runner that initialized JAX and installed signal hooks. Required missing constants still fail; unsupported AST expressions are skipped by the reader. Keep these comparisons substantive.

**Known minimal-CPU dependency boundary issue:** `src/fpcode/kpms/checkpoints.py` still eagerly imports `keypoint_moseq` and retains substantive `compact_checkpoint`/`is_checkpoint_packed` implementations. Newly discovered `tests/regression/test_wired_modules.py::test_checkpoint_compaction_equivalence` imports that module directly. The final lab-environment run emitted a TensorFlow Probability/JAX warning; CPU CI deliberately excludes this backend. Separate backend loading while retaining meaningful synthetic coverage. Preserve the old signatures, return dictionaries, `full`/`slim` semantics and historical shim identity. The new `checkpoint_io.compact_kpms_checkpoint` has a different interface and packing mode: it cannot simply replace the old function by name. A pass in the full lab environment is insufficient proof of clean CPU CI.

Moved manual material is not universally repaired. In particular:

- `studies/photometry_exploration/validation/run_oft_batch.sh` retains an old notebook-relative path, `runs` output, relative model directory and historical absolute test-cache input. Use explicit notebook/data/model/output parameters before running it.
- `studies/multiview_experiments/validation/test_flipping.py` omits the now-required image width.
- `test_flipping_width.py` prints some results instead of asserting them; utility paths in these scripts and `verify_flipping_invariance.py` need checking after the moves.

### Lint debt must be grounded in the old checkout

`ops/repo/lint.py` implements strict lint for `src/fpcode`, `pipelines`, backup and selected repository tooling; it also inventories all maintained studies/tests/ops diagnostics by path, rule, message and count. Ruff is pinned to 0.16.10. Line shifts do not create or hide debt.

**`registry/LINT_BASELINE.json` does not exist.** `make lint`, the new pre-commit lint hook and CI therefore remain unfinished. The last library Ruff check had 76 errors, 64 marked fixable; some predate this session, some are introduced ordering/unused imports. Strict paths must pass completely before the debt guard can run.

Do not blindly run `ruff --fix`. F401 currently flags intentional checkpoint reexports in `compat/io.py`; removing those imports would delete public APIs. Express intentional reexports with `as`/`__all__` and apply fixes in reviewed scopes. Other facades can have similar requirements. `common/file_io.py` also retains unused `List`/`Optional`; migrated tracking/evaluation/cache docstrings still mention `core`, and KPMS preprocessing documentation still names `fpcode.kpms.data`.

The preserved prior inventory has 3,254 diagnostics from 709 tracked Python files in **studies, tests and ops** at HEAD, including previously excluded loose tests, under the current Ruff rules. It excludes `src/fpcode` and `pipelines`; strict-path errors require repairs and, where necessary, a separate comparison against HEAD. Map the maintained-scope fingerprints through both move ledgers and the ten ops relocations. Account for changed module names inside diagnostic messages. Repair unmatched newly introduced diagnostics, then write/review only legitimate inherited debt. **Do not accept every current error with `make lint-baseline` and call it pre-existing.** CI must never automatically refresh that inventory.

### Packaging and CI: scaffold present, full reproducibility unproven

- `pyproject.toml` packages `fpcode/py.typed` and all three compatibility calibration JSON resources and installs five CLI entry points including `fpcode-export`.
- The test extra adds Pillow, OpenCV headless, imageio and imageio-ffmpeg. The optional typing extra adds mypy/pandas stubs with an initial contracts-only scope. Mypy and pre-commit were not executed; `py.typed` does not certify complete typing.
- The `all` extra still duplicates `kpms`; refactor it without changing dependency selection and verify metadata. Keep the legitimate distribution/import names.
- `configs/environments/ci-cpu-requirements.txt` pins 47 observed CPU/media distributions. These PyPI wheels were not independently installed. `pytest-timeout` is in the test extra but was not installed/required in the observed lab environment and is not in this CPU file.
- `.github/workflows/cpu-checks.yml` defines Ubuntu 24.04/Python 3.10.19 functional, architecture, empirical collection, lint and installed-wheel gates. It has not run on GitHub.
- `ops/repo/package_check.py` builds from a temporary source-only copy, installs offline into a temporary venv with existing scientific dependencies, verifies the installed package origin, resources and type marker, and runs all five CLI help commands outside the checkout without source `PYTHONPATH`.

That wheel check passed in 3.54 seconds before the final stop. It is useful evidence against source-checkout masking, **but it reused available lab dependencies**. It is not proof of installation into a minimal clean environment. Recheck after repairs, and validate the pinned CPU environment separately without JAX/KPMS.

## 10. Ignored source, documentation and historical evidence traps

The existing `.gitignore` has a blanket `/ops/diagnostics/*` rule except for two files. Five locally repaired maintained scripts are still ignored:

```text
ops/diagnostics/check_sampling_rates.py
ops/diagnostics/check_nframes_match.py
ops/diagnostics/generate_dual_centroid_snapshots.py
ops/diagnostics/verify_1h_parity.py
ops/diagnostics/generate_verification_evidence.py
```

Add deliberate source exceptions or explicitly stage reviewed source later. Directories named `data/` inside source/compatibility trees can also be ignored. Use `git check-ignore -v PATH` when a moved source file seems absent. Do not relax data ignore rules so that empirical binaries are accidentally committed.

Maintained README/index/recovery/report wording was corrected. `docs/ARCHITECTURE.md` retains a historical banner. Existing ZIPs under root `reports/` still represent the earlier checkout; none were regenerated. The updated future archive builder writes `archive/bundles`, includes small ledgers and symlink metadata, and scopes evidence claims. Physically archiving existing bundles is pending and should follow a reference audit.

No `AGENTS.md` was found in the repo/parents at the checkpoint. `.agents/rules/nomenclature.md` applies: use objective physical/kinematic/statistical descriptors for unsupervised components, not inferred subjective behavior labels. Its old `syllable_generation/...` guideline link is stale; the retained guideline is in `studies/language_models/mds/01_theory_and_strategy/04_objective_ethological_nomenclature_guidelines.md`. Do not introduce scientifically stronger labels during cleanup.

### Existing backup and GPU evidence have narrow scope

The earlier snapshot at `/mnt/hdd/backups/FiberphotometryCode/20261011T143033Z-snapshot-353ab324` was checked at the metadata/manifest level: 96,373,603,551 file bytes, 8,440 files, 1,316 directories and 330 links; Git HEAD recorded `e7010dd`; Git bundle size 702,166,666 bytes; manifest hash matched; sealed metadata mode 0555. Its policy excluded 1,730 paths totaling 1,070,605,488,639 bytes, including large files/videos/caches/outputs.

It predates this session's uncommitted moves and edits. No new snapshot, complete payload rehash or restore drill was performed here. Do not claim complete empirical recoverability or use it as proof that the current working tree is backed up.

Earlier CUDA evidence is retained at `outputs/20261011_142618-two-session-validation-7d681004/validation.json`: two sessions of 256 valid frames, two AR and two SLDS updates, CUDA device 0, about 24.98 seconds. It was not rerun after these changes. It does not prove full-cohort convergence, universal scientific equivalence or every historical pickle's compatibility. No untrusted empirical pickle was opened during this follow-up.

## 11. Ordered continuation plan

Work in small reviewable batches. Do not start with another whole-tree migration script.

1. **Inventory and preserve the interrupted tree.** Read both move ledgers, the compact checkpoint and current Git status. Confirm that moved source destinations exist, enumerate ignored source deliberately, and preserve current uncommitted files before any further structural changes. Do not interpret the 2,173 deleted status entries as disposable material.
2. **Repair the two newly broken child aliases.** Verify sibling targets first. Leave the unresolved historical guide as an explicit investigation item. Recheck symlinks without following external directory targets.
3. **Restore canonical/historical identity.** Retarget Ulrik/Datta shims and internal imports, then run the three import-order regressions. Reconcile the registry and updater in a bounded separate patch; preserve physical script and pickle obligations.
4. **Repair golden-source relocation lookup.** Keep original digests and 73-body coverage. Run the targeted four regressions, then the default functional suite and architecture gate.
5. **Finish consumer paths project by project.** Review helper previews/diffs, paths constructed dynamically, notebooks and shell launchers. Separate input existence from intentionally new output locations. Test root overrides in fresh processes. Do not delete transition aliases just because text search is cleaner.
6. **Extract shared session/movie mechanisms.** Introduce explicit configuration and callbacks, preserve policy ownership and the numerical/rendering distinctions above, and use independent small regression expectations. Handle observed behavioral bugs in separately justified changes.
7. **Finish remaining library integration.** Alias `pipelines/export.py`, split KPMS parsing/counting where appropriate, repair fusion dependency fingerprints, review historical profile access and fix migrated docstrings. Preserve API reexports and facade/module identity. Run relevant targeted checks as each change lands.
8. **Finish lint/typing/CI/packaging.** Repair strict lint, map inherited debt honestly, review/write the inventory, verify empirical explicit-path guards, resolve the synthetic backend dependency, validate a clean CPU environment and installed wheel outside the checkout. Run mypy only in a separate suitable environment if needed.
9. **Finish documentation and retirement.** Report actual validation and remaining limitations. Archive frozen bundles after checking references. Retire temporary aliases only once all import/filesystem/direct-script/pickle obligations are covered; do not invent a root-directory-count requirement.
10. **Review and stage source deliberately.** Ensure ignored diagnostics and moved source are included, while data/model/output payloads remain excluded. Inspect staged rename/add/delete relationships and tests. Commit coherent tested batches according to the user's continuation instructions. This handoff does not authorize an automatic merge or production maintenance operation.

### Core final validation commands

Run once the relevant repairs are complete; broaden testing when new changes or unresolved failures justify it. Commands below do not intentionally run empirical test bodies:

```bash
cd /home/fsp585/Desktop/FiberphotometryCode
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
FPCODE_PY=/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python

"$FPCODE_PY" -B -m pytest -q -p no:cacheprovider
"$FPCODE_PY" -B ops/repo/dependencies.py
"$FPCODE_PY" -B -m pytest tests/empirical --empirical --collect-only \
  -q -m 'needs_data or gpu or slow' -p no:cacheprovider
make lint PYTHON="$FPCODE_PY"
make package-check PYTHON="$FPCODE_PY"
```

Separately verify that an explicit empirical file without `--empirical` returns the intended usage error before imports, and repeat collection with a backend-blocking import guard. `make typecheck` is optional initial typing work and requires mypy; it was not a completed checkpoint gate.

A green architecture audit only measures configured static imports and cycles; computed dynamic imports and real path semantics still need review. A passing synthetic suite only covers its assertions. Record the exact final working tree/commit and environment with results, and report material remaining limitations plainly.

## 12. Completion checklist

- [ ] Current move destinations and source staging reconciled; no empirical payload accidentally committed.
- [ ] Two introduced dangling aliases repaired; historical missing guide resolved or explicitly documented.
- [ ] Historical/canonical identity and both import orders pass without eager config/backends.
- [ ] Original golden AST digests remain intact with 73 bodies checked at current implementation locations.
- [ ] Remaining maintained study/path/launcher consumers use explicit canonical owners and root semantics.
- [ ] Shared session/movie extraction completed and meaningfully tested, with study policy kept explicit.
- [ ] Historical export CLI forwards correctly; installed compaction help describes actual behavior.
- [ ] KPMS counting/parsing ownership and remaining library/docstring work assessed and completed as scoped.
- [ ] SHIMS registry/generator reflect the real tree and preserve physical execution and identity obligations.
- [ ] Strict lint passes; inherited diagnostic inventory is reviewed against the unchanged HEAD snapshot.
- [ ] Default suite, architecture, empirical collection guards and installed-wheel checks pass after final changes.
- [ ] CPU workflow dependencies verified without the modeling backend; remote CI result recorded when run.
- [ ] Maintained docs describe actual outcomes; older ZIPs/reports and backup/GPU limitations stay clearly dated.
- [ ] Temporary aliases retired only where compatibility obligations have been verified.

At handoff, these completion boxes are intentionally unchecked. Individual completed foundations are described above; **the overall refactor follow-up is not finished**.
