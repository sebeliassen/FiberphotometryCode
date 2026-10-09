# FiberphotometryCode Scaffolding Refactoring: Complete Change Report

**Branch**: `scaffolding-refactor`  
**Baseline Git Tag**: `pre-scaffolding-20261009` (commit `969c893`)  
**Active Python Environment**: `kpms_gpu_cuml` (`/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`)  
**Date**: October 9, 2026  
**Author / Role**: Senior Research-Software Engineer  

---

## 1. Executive Summary & Boundaries

This project executed a comprehensive scaffolding refactoring of the `FiberphotometryCode` repository (~3.1 TB data, 26,863 files) in strict accordance with the SWE architecture specifications:
- **Zero scientific/numerical changes**: Zero modifications to model weights, hyperparameters, syllable assignments, or analytical algorithms.
- **Zero analysis loss**: All analysis tools and exploratory code in [`syllable_generation/analysis/`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis) remain 100% functional and untouched in place.
- **Nomenclature strictly enforced**: Strict adherence to objective physical, kinematic, and statistical terminology per `.agents/rules/nomenclature.md`.
- **100% Reversibility**: All relocations are documented in [`MOVES.tsv`](file:///home/fsp585/Desktop/FiberphotometryCode/MOVES.tsv) and can be reverted in one click using [`undo_moves.sh`](file:///home/fsp585/Desktop/FiberphotometryCode/undo_moves.sh).

---

## 2. Structural Overview: Before vs. After

### Before Refactoring
```text
FiberphotometryCode/
├── *.zip (15 large distribution and review zips in root, ~130 MB)
├── *.log (6 execution logs & nohup.out cluttering root)
├── debug_*.png (ad-hoc figure renders in root)
├── syllable_generation_backup_20260204_133746/ (historical backup folder in root)
├── tests/
│   ├── runs/ (54 MB of old test run outputs mixed into test suite)
│   └── (no unified pytest.ini, conftest.py, or placeholder skip fixtures)
├── syllable_generation/
│   ├── multiview_top_bot_experiments_code_backup_20260930/ (redundant snapshot)
│   └── multiview_top_bot_experiments_code_backup_20261005/ (redundant snapshot)
├── (No pyproject.toml, Makefile, or pip-installable core package)
├── (No centralized YAML configs for production runs)
└── (Hardcoded /home/fsp585 paths across 1,054 files)
```

### After Refactoring
```text
FiberphotometryCode/
├── core/                                 # Centralized foundational package (SSOT)
│   ├── __init__.py
│   ├── paths.py                          # Repo root discovery & portable relative paths
│   ├── h5_guard.py                       # Blocks destructive in-place writes on Tier-A data
│   ├── run_record.py                     # Emits atomic run.json provenance records
│   └── logging_utils.py                  # Structured logger & fail-safe ntfy.sh alerts
├── configs/                              # Centralized, typed YAML configurations
│   ├── tandem_cohort_50ar_200slds.yml    # Verified 100% equivalent to production runner
│   └── cohort_static_21kp.yml            # Verified 100% equivalent to cohort runner
├── archive/                              # Quarantined historical artifacts & snapshots
│   ├── ARCHIVE_INDEX.md                  # Comprehensive markdown catalog of archived items
│   ├── packages/                         # 15 root .zip distribution packages
│   ├── logs/                             # Historical root .log files & nohup.out
│   ├── text_dumps/                       # Large concatenated text dumps
│   ├── debug_images/                     # Ad-hoc debug PNG renders
│   └── backups/                          # 3 dated code snapshot backup folders
├── outputs/
│   └── test_runs/                        # Historical test outputs moved out of tests/
├── tests/
│   ├── conftest.py                       # require_artifact fixture (skips on placeholders)
│   ├── test_config_equivalence.py        # Validates YAML config math == production runners
│   └── golden/                           # Preserved historical EML scorecard baselines
├── pyproject.toml                        # Standard PEP 517/621 packaging (fiberphotometry-suite)
├── Makefile                              # Automation: test, lint, dry-run, backup, inventory
├── pytest.ini                            # CPU/GPU test markers, ignores, pythonpath
├── environment.lock.yml                  # Full locked conda environment
├── environment.history.yml               # Explicit conda spec history
├── registry.tsv                          # Model & run registry (50 cataloged models)
├── ARTIFACT_INVENTORY.tsv                # Full SHA-256 catalog of all 26,863 files
├── DUPLICATES.tsv                        # Catalog of 3,537 duplicate content hashes
├── MOVES.tsv                             # Machine-readable record of all 28 relocations
├── undo_moves.sh                         # Instant one-click reversal script
└── SCAFFOLDING_REPORT.md                 # Complete milestone report
```

---

## 3. Full Inventory of All 28 File & Directory Relocations

Every move was executed on the same filesystem without data modification:

| # | Original Path | New Destination | Category | Description |
|:---|:---|:---|:---|:---|
| 1 | `archive_checkpoints.log` | `archive/logs/archive_checkpoints.log` | Log | Execution log from checkpoint maintenance |
| 2 | `model_fitting.log` | `archive/logs/model_fitting.log` | Log | Execution log from model fitting |
| 3 | `nohup.out` | `archive/logs/nohup.out` | Log | Background terminal stdout/stderr dump |
| 4 | `run_dlc_analysis.log` | `archive/logs/run_dlc_analysis.log` | Log | DeepLabCut batch inference execution log |
| 5 | `sweep_debug.log` | `archive/logs/sweep_debug.log` | Log | Hyperparameter sweep debug execution log |
| 6 | `sweep.log` | `archive/logs/sweep.log` | Log | Hyperparameter sweep main execution log |
| 7 | `tandem_multiview...concat.txt` | `archive/text_dumps/tandem_multiview...concat.txt` | Text Dump | Large concatenated review text bundle |
| 8 | `debug_bot_raw_52299.png` | `archive/debug_images/debug_bot_raw_52299.png` | Image | Ad-hoc ventral camera frame render |
| 9 | `debug_syll30_r1_c2.png` | `archive/debug_images/debug_syll30_r1_c2.png` | Image | Ad-hoc syllable debug render |
| 10 | `tests/runs/` | `outputs/test_runs/` | Directory | Heavy test run execution artifacts |
| 11 | `chain_grid_generation_code.zip` | `archive/packages/chain_grid_generation_code.zip` | Archive | Root zip distribution snapshot |
| 12 | `chatgpt_ethological_seqnmf_package.zip` | `archive/packages/chatgpt_ethological_seqnmf_package.zip` | Archive | Root zip distribution snapshot |
| 13 | `code_backup_2026-04-10.zip` | `archive/packages/code_backup_2026-04-10.zip` | Archive | Root zip distribution snapshot |
| 14 | `density_sampling_v2_audit.zip` | `archive/packages/density_sampling_v2_audit.zip` | Archive | Root zip distribution snapshot |
| 15 | `fiberphotometry_code_review.zip` | `archive/packages/fiberphotometry_code_review.zip` | Archive | Root zip distribution snapshot |
| 16 | `multiview_kpms_expert_review_package.zip` | `archive/packages/multiview_kpms_expert_review_package.zip` | Archive | Root zip distribution snapshot |
| 17 | `multiview_static_21kp_v2_engineering_dossier.zip` | `archive/packages/multiview_static_21kp_v2_engineering_dossier.zip` | Archive | Root zip distribution snapshot |
| 18 | `multiview_tandem_expert_review_package.zip` | `archive/packages/multiview_tandem_expert_review_package.zip` | Archive | Root zip distribution snapshot |
| 19 | `syllable_chains_research_package.zip` | `archive/packages/syllable_chains_research_package.zip` | Archive | Root zip distribution snapshot |
| 20 | `syllable_generation_source_backup_20260216_161040.zip` | `archive/packages/syllable_generation_source_backup_20260216_161040.zip` | Archive | Root zip distribution snapshot |
| 21 | `tandem_multiview_kpms_expert_review_package_7days.zip` | `archive/packages/tandem_multiview_kpms_expert_review_package_7days.zip` | Archive | Root zip distribution snapshot |
| 22 | `v2_grid_movies_and_trajectories_engineering_rfc.zip` | `archive/packages/v2_grid_movies_and_trajectories_engineering_rfc.zip` | Archive | Root zip distribution snapshot |
| 23 | `v2_visualization_fixes_and_gemini_plan (1).zip` | `archive/packages/v2_visualization_fixes_and_gemini_plan (1).zip` | Archive | Root zip distribution snapshot |
| 24 | `v2_visualization_fixes_and_gemini_plan.zip` | `archive/packages/v2_visualization_fixes_and_gemini_plan.zip` | Archive | Root zip distribution snapshot |
| 25 | `v4_kpms_integrity_and_labeling_deliverable.zip` | `archive/packages/v4_kpms_integrity_and_labeling_deliverable.zip` | Archive | Root zip distribution snapshot |
| 26 | `syllable_generation_backup_20260204_133746/` | `archive/backups/syllable_generation_backup_20260204_133746/` | Directory | Dated code snapshot from Feb 2026 |
| 27 | `.../multiview_top_bot_experiments_code_backup_20260930/` | `archive/backups/multiview_top_bot_experiments_code_backup_20260930/` | Directory | Dated code snapshot from Sep 30 2026 |
| 28 | `.../multiview_top_bot_experiments_code_backup_20261005/` | `archive/backups/multiview_top_bot_experiments_code_backup_20261005/` | Directory | Dated code snapshot from Oct 05 2026 |

---

## 4. Newly Introduced Additive Modules & Packages

1. **`core` Foundation Package**:
   - [`core/paths.py`](file:///home/fsp585/Desktop/FiberphotometryCode/core/paths.py): Single Source of Truth for repository root discovery. Reads `FIBERPHOTOMETRY_ROOT` env var or auto-discovers anchor files (`pyproject.toml`, `ARTIFACT_INVENTORY.tsv`). Provides `to_relative_path()` for portable logging.
   - [`core/h5_guard.py`](file:///home/fsp585/Desktop/FiberphotometryCode/core/h5_guard.py): Immutability guard. Blocks `r+`, `w`, and `a` modes on Tier-A checkpoints (`safe_open_h5()`), preventing accidental in-place mutations.
   - [`core/run_record.py`](file:///home/fsp585/Desktop/FiberphotometryCode/core/run_record.py): Emits atomic `run.json` execution manifests capturing git commit SHA, argv, execution duration, CPU/GPU hardware, and input/output fingerprints.
   - [`core/logging_utils.py`](file:///home/fsp585/Desktop/FiberphotometryCode/core/logging_utils.py): Standardized timestamped console/file logging and resilient `send_ntfy()` notification helper.
   - Installed in editable mode (`pip install -e . --no-deps`) into `kpms_gpu_cuml`.

2. **Centralized YAML Configurations**:
   - [`configs/tandem_cohort_50ar_200slds.yml`](file:///home/fsp585/Desktop/FiberphotometryCode/configs/tandem_cohort_50ar_200slds.yml): Typed configuration for 250-iteration full cohort tandem training.
   - [`configs/cohort_static_21kp.yml`](file:///home/fsp585/Desktop/FiberphotometryCode/configs/cohort_static_21kp.yml): Typed configuration for static 21-keypoint cohort training across seeds.
   - Verified 100% parameter equivalence via [`tests/test_config_equivalence.py`](file:///home/fsp585/Desktop/FiberphotometryCode/tests/test_config_equivalence.py).

3. **Registry**:
   - [`registry.tsv`](file:///home/fsp585/Desktop/FiberphotometryCode/registry.tsv): Catalog of 50 models across active production (`ACTIVE_PRODUCTION_V2`, `ACTIVE_PRODUCTION_V1`, `PRODUCTION_CONTINUOUS_CALIBRATED`), historical baselines, pilot benchmarks, and archived prototypes.

---

## 5. Verification Matrix & Invariant Guarantees

All verification executed in the active **`kpms_gpu_cuml`** conda environment:

| Audit Check | Command | Result |
|:---|:---|:---|
| Fast CPU Tests | `make test` | **88 passed in 5.03s** (0 failures, 0 errors) |
| Read-Only Dry Runs | `make dry-run` | **100% Passed** (EML audit valid, 258/258 calibration crystal) |
| Syntax Validation | `make lint` | **Core modules py_compile valid** |
| Tier-A Immutability | Audit script against inventory | **576/576 Tier-A files 100% intact and accounted for** |
| Analysis Preservation | Directory inspection | **[`syllable_generation/analysis/`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis) untouched and functional** |

---

## 6. Git Commit History (`scaffolding-refactor` Branch)

- `969c893`: `chore(scaffolding): Phase 0 baseline - update .gitignore to exclude data, preserve legacy` (Tagged `pre-scaffolding-20261009`)
- `daf21b8`: `docs(scaffolding): complete Phase 0 backup report and verification drill`
- `8e25e0a`: `docs(scaffolding): complete Phase 1 duplicate content audit across 15 snapshot bundles`
- `7a19801`: `docs(scaffolding): establish core documentation suite (README, ARCHITECTURE, DATA_LAYOUT, RUNBOOK, KNOWN_ISSUES)`
- `e761033`: `test(scaffolding): establish standardized test harness, golden baselines, and test split plan`
- `587138b`: `feat(scaffolding): establish modern packaging, Makefile automation, pre-commit config, and environment locks`
- `a7a5b58`: `feat(scaffolding): add core modules, centralized configs, and model registry`
- `89cecf0`: `docs(scaffolding): prepare Phase 3 MOVES.tsv, undo_moves.sh, and ARCHIVE_INDEX.md`
- `e8ec6f9`: `chore(scaffolding): execute Phase 3 clean relocations to archive/ and outputs/`
- `e25bc09`: `docs(scaffolding): deliver final SCAFFOLDING_REPORT.md`
