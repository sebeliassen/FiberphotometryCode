# Comprehensive Repository Scaffolding Report

**Repository**: `FiberphotometryCode`  
**Execution Date**: October 9, 2026  
**Environment**: `kpms_gpu_cuml` (`/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`)  
**Git Branch**: `scaffolding-refactor`  
**Initial Baseline Tag**: `pre-scaffolding-20261009`  

---

## 1. Executive Summary & Boundaries

This project executed a comprehensive scaffolding refactoring of the `FiberphotometryCode` repository (~3.1 TB data, 26,863 files) in strict accordance with the SWE architecture specifications:
- **Scope**: Layout organization, test harness, configuration management, execution provenance, packaging, documentation, and backup guarantees.
- **Strict Constraint**: **Zero changes to science, numerics, labels, or outputs.** All analysis pipelines and exploratory code in [`syllable_generation/analysis/`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis) were preserved in place without modification.
- **Nomenclature Compliance**: Strict adherence to objective physical, kinematic, and statistical terminology per `.agents/rules/nomenclature.md`.

---

## 2. Phase 0: Physical Backup & Immutability Protection

1. **Standalone Snapshot**:
   - Packaged standalone `code_snapshot.tar.gz` (127.94 MB) with full SHA-256 manifest to `/home/fsp585/Desktop/_backups/20261009_112705/` and `/mnt/hdd/backups/code_snapshots/20261009_112705/`.
2. **Artifact Inventory**:
   - Generated [`ARTIFACT_INVENTORY.tsv`](file:///home/fsp585/Desktop/FiberphotometryCode/ARTIFACT_INVENTORY.tsv) (5.60 MB, 26,863 files) cataloging paths, sizes, mtimes, tiers, and checksums.
3. **Physical Secondary Disk Backup**:
   - 509 Tier-A files (< 2 GB, 44.22 GB total) copied to `/mnt/hdd/backups/TierA_backup_20261009/` with 100% hash verification.
4. **In-Place Write Protection**:
   - Write-protected 67 massive checkpoints ($\ge$ 2 GB, 861.17 GB) via `chmod a-w`. Recorded in [`PERMISSIONS_MODIFIED_TIER_A.tsv`](file:///home/fsp585/Desktop/FiberphotometryCode/PERMISSIONS_MODIFIED_TIER_A.tsv), with an instant restoration script [`restore_tier_a_permissions.sh`](file:///home/fsp585/Desktop/FiberphotometryCode/restore_tier_a_permissions.sh).
5. **Restoration Drill**:
   - Byte-for-byte restore drill verified on `kpms_engineering_handoff_v2/SOURCE_MANIFEST.json`.

---

## 3. Phase 1: Audit, Documentation & Test Harness

1. **Deduplication Audit**:
   - Cataloged 3,537 duplicate content hashes in [`DUPLICATES.tsv`](file:///home/fsp585/Desktop/FiberphotometryCode/DUPLICATES.tsv). 15 diverged review bundles marked `DIVERGED (TOUCH NOTHING)`.
2. **Standard Documentation**:
   - Created clean root [`README.md`](file:///home/fsp585/Desktop/FiberphotometryCode/README.md).
   - Created [`docs/ARCHITECTURE.md`](file:///home/fsp585/Desktop/FiberphotometryCode/docs/ARCHITECTURE.md) (system design, pipelines, invariants).
   - Created [`docs/DATA_LAYOUT.md`](file:///home/fsp585/Desktop/FiberphotometryCode/docs/DATA_LAYOUT.md) (storage tiers, disk quotas).
   - Created [`docs/RUNBOOK.md`](file:///home/fsp585/Desktop/FiberphotometryCode/docs/RUNBOOK.md) (daily operational commands, backup management).
   - Created [`docs/KNOWN_ISSUES.md`](file:///home/fsp585/Desktop/FiberphotometryCode/docs/KNOWN_ISSUES.md) (formal ledger of known historical bugs, marked `[NOT FIXED]`).
3. **Test Infrastructure**:
   - Standardized [`pytest.ini`](file:///home/fsp585/Desktop/FiberphotometryCode/pytest.ini) with CPU/GPU markers and data exclusion filters.
   - Built [`tests/conftest.py`](file:///home/fsp585/Desktop/FiberphotometryCode/tests/conftest.py) with `require_artifact()` fixture ensuring tests skip gracefully on trimmed archives.
   - Preserved separate golden evaluation baselines in [`tests/golden/`](file:///home/fsp585/Desktop/FiberphotometryCode/tests/golden/).

---

## 4. Phase 2: Additive Scaffolding & Packaging

1. **Packaging**:
   - Defined [`pyproject.toml`](file:///home/fsp585/Desktop/FiberphotometryCode/pyproject.toml) and installed editable package `fiberphotometry-suite` into `kpms_gpu_cuml`.
   - Created [`Makefile`](file:///home/fsp585/Desktop/FiberphotometryCode/Makefile) automating `make test`, `make lint`, `make dry-run`, and backup management.
   - Locked conda environments in [`environment.lock.yml`](file:///home/fsp585/Desktop/FiberphotometryCode/environment.lock.yml) and [`environment.history.yml`](file:///home/fsp585/Desktop/FiberphotometryCode/environment.history.yml).
2. **Core SSOT Library (`core/`)**:
   - [`core/paths.py`](file:///home/fsp585/Desktop/FiberphotometryCode/core/paths.py): Centralized root discovery, removing hardcoded paths.
   - [`core/h5_guard.py`](file:///home/fsp585/Desktop/FiberphotometryCode/core/h5_guard.py): Defends against destructive in-place HDF5 writes.
   - [`core/run_record.py`](file:///home/fsp585/Desktop/FiberphotometryCode/core/run_record.py): Emits atomic `run.json` provenance records.
   - [`core/logging_utils.py`](file:///home/fsp585/Desktop/FiberphotometryCode/core/logging_utils.py): Centralized logging and fail-safe ntfy alerts.
3. **Centralized Configurations & Equivalence Proof**:
   - Created typed YAML configurations: [`configs/tandem_cohort_50ar_200slds.yml`](file:///home/fsp585/Desktop/FiberphotometryCode/configs/tandem_cohort_50ar_200slds.yml) and [`configs/cohort_static_21kp.yml`](file:///home/fsp585/Desktop/FiberphotometryCode/configs/cohort_static_21kp.yml).
   - Validated parameter equivalence via [`tests/test_config_equivalence.py`](file:///home/fsp585/Desktop/FiberphotometryCode/tests/test_config_equivalence.py).
4. **Model Registry**:
   - Cataloged 50 model checkpoints across active production, baselines, and prototypes in [`registry.tsv`](file:///home/fsp585/Desktop/FiberphotometryCode/registry.tsv).

---

## 5. Phase 3: Clean Reorganization & Reversibility Guarantee

1. **Clean Moves Executed**:
   - All 15 root distribution `.zip` archives relocated to `archive/packages/`.
   - Root execution logs and stdout dumps relocated to `archive/logs/` and `archive/text_dumps/`.
   - Ad-hoc debug images relocated to `archive/debug_images/`.
   - Historical test execution dumps (`tests/runs/`) relocated to `outputs/test_runs/`.
   - Dated code backup folders (`syllable_generation_backup_20260204_133746/`, `multiview_top_bot_experiments_code_backup_*`) relocated to `archive/backups/`.
2. **Reversibility**:
   - Full catalog documented in [`MOVES.tsv`](file:///home/fsp585/Desktop/FiberphotometryCode/MOVES.tsv) and [`archive/ARCHIVE_INDEX.md`](file:///home/fsp585/Desktop/FiberphotometryCode/archive/ARCHIVE_INDEX.md).
   - Executable script [`undo_moves.sh`](file:///home/fsp585/Desktop/FiberphotometryCode/undo_moves.sh) tested and ready to revert all moves instantly if needed.

---

## 6. Verification & Quality Gates

| Verification Target | Command / Check | Result |
|:---|:---|:---|
| Fast CPU Test Suite | `make test` | **88 passed** (5.03s, 0 failures) |
| Read-Only Dry Runs | `make dry-run` | **100% Passed** (EML audit + 258/258 calibration crystal) |
| Syntax Validation | `make lint` | **Core modules py_compile valid** |
| Tier-A Immutability | Audit script against inventory | **576/576 Tier-A files 100% accounted for and intact** |
| Analysis Preservation | Working tree inspection | **`syllable_generation/analysis/` untouched** |
