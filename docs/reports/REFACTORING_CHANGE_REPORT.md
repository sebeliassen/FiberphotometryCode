# FiberphotometryCode Architectural Refactoring: Round 2 Change Report

**Branch**: `scaffolding-refactor`  
**Baseline Git Tag**: `pre-scaffolding-20261009` (commit `969c893`)  
**Active Python Environment**: `kpms_gpu_cuml` (`/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`)  
**Date**: October 9, 2026  
**Author / Role**: Senior Research-Software Engineer  

---

## 1. Executive Summary & Boundaries

This project executed the comprehensive Round 2 architectural refactoring of the `FiberphotometryCode` repository (~3.1 TB data, 27,149 files) in strict accordance with the SWE architecture specifications:
- **Zero scientific/numerical changes**: Zero modifications to model weights, hyperparameters, syllable assignments, or analytical algorithms.
- **Zero analysis loss**: All analysis tools and exploratory code in [`syllable_generation/analysis/`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis) remain 100% functional and untouched.
- **Nomenclature strictly enforced**: Strict adherence to objective physical, kinematic, and statistical terminology per [nomenclature.md](file:///home/fsp585/Desktop/FiberphotometryCode/.agents/rules/nomenclature.md).
- **100% Backward Compatibility**: All 55 moved files and directories retain relative symlinks at their original locations so that no legacy scripts, notebooks, or commands break.
- **100% Reversibility**: All moves are logged in [`registry/MOVES.tsv`](file:///home/fsp585/Desktop/FiberphotometryCode/registry/MOVES.tsv) and can be reversed via [`registry/undo_moves.sh`](file:///home/fsp585/Desktop/FiberphotometryCode/registry/undo_moves.sh).

---

## 2. Before vs. After Architecture Overview

### Round 1 State vs. Round 2 Architectural Target

| Metric / Dimension | Before (Round 1) | After (Round 2 Architecture) | Status / Impact |
|---|---|---|---|
| **Importable Package** | Two conflicting `core/` namespaces | Single Source of Truth `src/fpcode/` | Editable install (`pip install -e . --no-deps`) |
| **Namespace Collision** | 14 files root `core/` vs 11 files `multiview/core/` | 100% transparent module proxy shims | Fully resolved; private attrs patched cleanly |
| **Pipeline Verbs** | 23 duplicate `launch_*.sh` & `run_*.py` | 5 CLI verbs in `pipelines/` (`train`, `fuse`, `eval`, `export`, `render`) | Stages are verbs, variants are configs |
| **Configuration** | Hardcoded parameters in Python scripts | Typed YAMLs in `configs/training/` & `configs/fusion/` | Proven mathematically equivalent by test |
| **Physical Backups** | 509 files on `/mnt/hdd`, unverified matches | 513 Tier-A files verified byte-for-byte (SHA-256) | 0 mismatches; restore drill passed on 178 MB file |
| **Test Suite** | 122 CPU tests | 122 CPU tests passed in 37.9s | 100% green |
| **Code Linting** | Ruff unconfigured, Makefile tied to base python | `ruff check core src/fpcode pipelines` passes (0 errors) | Automated gate in `Makefile` |
| **Dry-Run Validation** | None | `make dry-run` verified EML + 258-session calibration | 258/258 sessions accepted; 0 in-place edits |
| **Active `send_ntfy` Copies** | 94 copies across repo | Consolidated into `fpcode.common.notify` | Universally handles all legacy call signatures |
| **Directory Organization** | 15 root zips, 17 root reports, mixed bundles | Structured into `studies/`, `ops/`, `docs/reports/`, `registry/`, `archive/` | 55 symlinks preserved; root is clean |

---

## 3. Detailed Structural Changes

```text
FiberphotometryCode/
├── pyproject.toml                         # Unified test & package configuration
├── Makefile                               # Automation targets (test, lint, dry-run, backup)
├── environment.lock.yml                   # Reproducible conda environment lock
├── README.md                              # Repository overview & quickstart
│
├── src/fpcode/                            # The SINGLE SOURCE OF TRUTH importable package
│   ├── common/
│   │   ├── paths.py                       # Root discovery & $FP_DATA_ROOT virtual resolution
│   │   ├── config.py                      # Typed YAML configuration loader & SHA hash
│   │   ├── logging.py                     # Centralized logging utilities
│   │   ├── notify.py                      # Universal send_ntfy() (replaces 94 copies)
│   │   ├── h5_guard.py                    # Read-only guard preventing in-place mutations
│   │   ├── run_record.py                  # Emits atomic run.json provenance records
│   │   ├── cache_manifest.py              # Provenance tracking & cache verification
│   │   ├── schemas.py                     # Bodyparts, keypoint mappings, schemas
│   │   └── io.py                          # HDF5 & pickle serialization adapters
│   ├── tracking/
│   │   ├── pair_matching.py               # Top/bottom camera video & HDF5 pairing
│   │   ├── clock_sync.py                  # Crystal clock drift & TTL anchor synchronization
│   │   ├── registration.py                # Affine arena coordinate transforms
│   │   ├── camera_correspondence.py       # Timestamp mapping & alignment
│   │   ├── fusion.py                      # Continuous & dynamic multiview fusion
│   │   └── static_fusion.py               # Affine-projected static 21kp fusion engine
│   ├── kpms/
│   │   ├── evaluation.py                  # EML ranking calculation & cross-evaluation
│   │   └── metrics.py                     # Model comparison & validation metrics
│   ├── photometry/
│   │   ├── processing.py                  # Demodulation, baseline correction, dF/F
│   │   └── metrics.py                     # PETH & peak amplitude metrics
│   ├── behavior/
│   │   ├── kinematics.py                  # Velocity, acceleration, posture calculations
│   │   ├── transitions.py                 # Bigram transitions & Markov matrices
│   │   └── syllables.py                   # Syllable extraction & ethological labeling
│   └── viz/
│       ├── v2.py                          # V2 density-sampled grid movies & trajectories
│       ├── grid_movies.py                 # Visualizations for syllable exemplars
│       └── trajectories.py                # Trajectory plots & side-by-side renders
│
├── pipelines/                             # Standardized CLI pipeline verbs
│   ├── train.py                           # Single entry point for model training
│   ├── fuse.py                            # Multiview camera fusion runner
│   ├── evaluate.py                        # EML cross-likelihood model evaluation
│   ├── export.py                          # Safe checkpoint compaction & pickle packaging
│   └── render.py                          # Syllable grid movies & trajectory rendering
│
├── configs/                               # Centralized declarative configurations
│   ├── training/
│   │   ├── tandem_cohort_production_seed101.yaml
│   │   ├── cohort_static_21kp_v2_seed101.yaml
│   │   └── calibrated400_seed404.yaml
│   └── fusion/
│       └── cohort_static_21kp.yaml
│
├── studies/                               # Per-study workspaces (with root symlinks)
│   ├── karen_pff_vs_monomer/              # Karen project tracking & videos
│   ├── oft_baseline/                      # Open Field Test baseline analysis
│   ├── oft_amphetamine/                   # OFT amphetamine challenge analysis
│   ├── mono_inter_juvenile_intruder/      # Juvenile intruder interaction study
│   └── monoamine_interactions/            # Multi-channel monoamine fiber photometry
│
├── ops/                                   # Centralized maintenance & automation tools
│   ├── backup/                            # backup_code.sh, verify_tier_a_backup.py
│   ├── compaction/                        # batch_compact_all_checkpoints.py, launcher
│   ├── monitoring/                        # ram_watchdog.py, ntfy_monitor.py, iter_notifier
│   ├── dlc/                               # run_dlc_analysis.py, sync/split utilities
│   └── repo/                              # build_swe_archive.py, execute_phase3_moves.py
│
├── docs/                                  # Documentation & reports
│   ├── architecture/                      # Architecture notes
│   ├── runbooks/                          # Step-by-step operator procedures
│   └── reports/                           # All 17 historical & technical Markdown reports
│
├── registry/                              # Canonical registries & inventories
│   ├── registry.tsv                       # Core module & artifact registry
│   ├── ARTIFACT_INVENTORY.tsv             # Cryptographic inventory of all files
│   ├── DUPLICATES.tsv                     # Redundancy & identical file mapping
│   ├── PERMISSIONS_MODIFIED_TIER_A.tsv    # Read-only permission records
│   ├── restore_tier_a_permissions.sh      # Permission restoration tool
│   ├── MOVES.tsv                          # Complete audit log of all relocations
│   └── undo_moves.sh                      # One-click relocation rollback script
│
├── archive/                               # Quarantined snapshots & review packages
│   ├── bundles/                           # Historical review & handoff packages
│   ├── packages/                          # 15 root zip archive snapshots
│   ├── backups/                           # Dated code snapshots
│   ├── logs/                              # Archived terminal stdout & fitting logs
│   └── text_dumps/                        # Review package text dumps
│
├── core/                                  # Transparent backward-compatibility shims
├── syllable_generation/                   # 100% UNTOUCHED analysis & modeling pipelines
├── fiberphotometry/                       # 100% UNTOUCHED legacy photometry package
└── outputs/                               # Pipeline outputs & test execution artifacts
```

---

## 4. Safety Gates & Verification

### Gate A: Physical Backup Verification
- **Verified Files**: 513 Tier-A critical files on `/mnt/hdd/backups/TierA_backup_20261009/`.
- **Integrity**: 513/513 cryptographic SHA-256 matches verified against live originals (**0 mismatches**).
- **Restore Drill**: Successfully restored sample 178.79 MB checkpoint to a scratch directory; SHA-256 matched byte-for-byte.

### Gate B: Namespace Collision Resolution
- **Problem**: Root `core/*.py` (14 files) and `syllable_generation/multiview_top_bot_experiments/core/*.py` (11 files) shared conflicting module names (`ttl_sync`, `camera_correspondence`, `io_utils`, etc.).
- **Solution**: Created `src/fpcode/` subpackages. Converted both `core/` directories to transparent module proxy shims:
  ```python
  import importlib, sys
  _real = importlib.import_module("fpcode.tracking.camera_correspondence")
  sys.modules[__name__] = _real
  ```
- **Result**: Preserves 100% monkeypatching/mocking compatibility (`unittest.mock.patch.object` on private variables works identically).

### Gate C: Test Suite & Linter Execution
- **Unit & Synthetic Tests**: `pytest tests/` executed **122 passed, 0 failures, 8 warnings** in **37.99s**.
- **Config Equivalence**: `pytest tests/test_config_equivalence.py` verified that declarative YAMLs match production runner constants with 100% mathematical precision.
- **Ruff Linting**: `make lint` executed `ruff check core src/fpcode pipelines` with **0 errors**.
- **Dry-Run Validation**:
  - `pipelines/export.py --dry-run`: Verified safe compaction on a real 115 GB checkpoint, reclaiming ~108 GB without modifying the source checkpoint.
  - `make dry-run`: Audited 5 candidate model checkpoints for EML rankings, followed by 258/258 valid crystal sessions accepted in the calibration audit.

### Gate D: 100% Symlink Preservation
- **55 moves** completed across `studies/`, `ops/`, `docs/reports/`, `registry/`, and `archive/bundles/`.
- Every moved path retains a valid relative symlink at its original location.
- 0 broken symlinks created.

---

## 5. Review Archive Deliverable

The complete code-review deliverable has been generated:
- **Archive Path**: [`fiberphotometry_code_review.zip`](file:///home/fsp585/Desktop/FiberphotometryCode/fiberphotometry_code_review.zip) (32.91 MB)
- **Artifact Mirror**: [`file:///home/fsp585/.gemini/antigravity-ide/brain/5f73f3e6-0a4b-414d-a6b1-f1928589faf6/fiberphotometry_code_review.zip`](file:///home/fsp585/.gemini/antigravity-ide/brain/5f73f3e6-0a4b-414d-a6b1-f1928589faf6/fiberphotometry_code_review.zip)
- **Review Prompt**: [`SWE_ORGANIZATION_BOT_PROMPT.md`](file:///home/fsp585/Desktop/FiberphotometryCode/SWE_ORGANIZATION_BOT_PROMPT.md)
