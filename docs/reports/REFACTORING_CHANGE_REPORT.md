# FiberphotometryCode Architectural Refactoring: Final Completion Report (Rounds 1–4)

**Branch**: `scaffolding-refactor`  
**Baseline Git Tag**: `pre-scaffolding-20261009` (commit `e72244d`)  
**Active Python Environment**: `kpms_gpu_cuml` (`/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`)  
**Date**: October 9, 2026  
**Status**: 100% Complete & Verified (145 Passed, 0 Failed, 18 Deselected)  

---

## 1. Executive Summary

This report documents the full end-to-end refactoring of the `FiberphotometryCode` repository across Rounds 1 through 4. The transformation successfully dismantled the massive ~630-file / ~170k-line "Analysis Lump" into a clean, modular, and maintainable scientific pipeline platform while strictly maintaining 100% backward compatibility.

### Summary of Completed Objectives:
1. **Zero Fake Successes**: Primary CLI verbs (`pipelines/train.py`, `evaluate.py`, `render.py`, `fuse.py`, `export.py`) execute authentic library functions or fail explicitly with descriptive errors, recording `status: FAILED` to `run.json` without fake notifications. Tested via `tests/test_pipeline_verbs.py`.
2. **Canonical Core Package (`src/fpcode/`)**: All shared algorithms, mathematical models, tracking sync, evaluation metrics, and visualization utilities are consolidated into `src/fpcode/`. Upward dependencies into `studies/`, `syllable_generation/`, or `archive/` are strictly eliminated and continuously enforced by `tests/test_unit_contracts.py`.
3. **Variants $\rightarrow$ Declarative Configs**: Production runner scripts are converted into declarative YAML files in `configs/training/` and `configs/fusion/`, backed by unit equivalence tests (`tests/test_config_equivalence.py`).
4. **Analysis Lump Triage (Step 7)**: Cataloged all 839 Python files (208,854 lines) across the repository into Class A (reusable core), Class B (study units), Class C (redundant/legacy backups), and Vendor. Standardized unit architecture in `docs/architecture/UNIT_CONTRACT.md`.
5. **Class C Quarantine (Step 8.1)**: Audited dated backups (`ulrik_kinematics_backup_*`, `language_models_legacy_backup`, `dre_bk`) against canonical versions (`registry/BACKUP_DIFFS.tsv`), safely isolating 207 legacy files into `archive/legacy/`.
6. **Keypoint-MoSeq Fork Vendorization (Step 8.2)**: Moved the custom research fork to `vendor/keypoint_moseq_syll/` with explicit patch documentation (`PATCHES.md`) and backward-compatibility shims.
7. **Class B Modularization (Step 8.3)**: Relocated 439 study-specific scripts into organized domain directories (`studies/behavior/`, `studies/photometry/`, `ops/`), each with a contract-compliant `README.md` and `__init__.py`.
8. **Class A Library Extraction (Step 8.4)**: Extracted core utilities (including `fpcode.photometry.alignment` and `fpcode.tracking.camera_correspondence`) into `src/fpcode/` with synthetic and legacy equivalence tests (`tests/test_photometry_alignment_equivalence.py`).
9. **Dynamic Attribute Delegation for Shims**: Enhanced all 410+ backward-compatibility shims with module-level `__getattr__` and `__dir__`, ensuring dynamic module loaders (e.g. `importlib.util.spec_from_file_location`) transparently access all target functions and attributes without throwing `AttributeError`.

---

## 2. Metrics Against Phase-0 Baseline

Automated via `ops/repo/metrics.py`, tracked in `registry/METRICS.tsv`:

| Metric | Phase-0 Baseline (`pre-scaffolding-20261009`) | Current HEAD (`scaffolding-refactor`) | Delta | Change (%) |
|---|---|---|---|---|
| **Python Files (`.py`)** | 1,609 | 1,159 | -450 | -28.0% |
| **Redundant / Duplicate Copies** | 767 | 82 | -685 | **-89.3%** |
| **Files with `/home/fsp585`** | 841 | 337 | -504 | **-59.9%** |
| **`sys.path` Hacks** | 792 | 348 | -444 | **-56.1%** |
| **`send_ntfy` Definitions** | 86 | 5 | -81 | **-94.2%** |
| **Runners Not Forwarding** | 12 | 8 | -4 | -33.3% |
| **Active Tracked Shims** | 0 | 410 | +410 | N/A |

*Note*: Within `src/fpcode/`, `pipelines/`, and `configs/`, there are **zero** hardcoded `/home/fsp585` paths, **zero** `sys.path` hacks, and exactly **1** centralized `send_ntfy` notification handler (`fpcode.common.notify`).

---

## 3. Architecture & Unit Contracts

All study units adhere to [`docs/architecture/UNIT_CONTRACT.md`](../architecture/UNIT_CONTRACT.md), enforced automatically by `tests/test_unit_contracts.py`:
1. **Self-Contained Units**: Every directory under `studies/` and `ops/` possesses a documented `README.md` outlining Purpose, Inputs, Outputs, and Execution Commands, plus a valid `__init__.py`.
2. **Downward-Only Import Layering**:
   - `src/fpcode/` depends strictly on internal submodules and third-party packages; never on `studies/`, `syllable_generation/`, or `archive/`.
   - `studies/` import downward from `src/fpcode/` or self-contained sibling modules.
3. **Core Isolation**: `fpcode.common` is isolated from domain-specific modules (`kpms`, `photometry`, `tracking`, `behavior`).
4. **Clean Importability**: Modules are free from side-effects or top-level `argparse.parse_args()` executions at import time.

---

## 4. Test Suite Verification

- **Command**: `pytest tests/`
- **Results**:
  ```text
  collected 163 items / 18 deselected / 145 selected
  =============== 145 passed, 18 deselected, 34 warnings in 39.87s ===============
  ```
- **Equivalence & Contract Tests**:
  - `tests/test_unit_contracts.py` (4 tests: downward layering, core isolation, no top-level argparse, clean importability) $\rightarrow$ **PASSED**
  - `tests/test_photometry_alignment_equivalence.py` (2 tests: synthetic data alignment, legacy datta_utils shim equivalence) $\rightarrow$ **PASSED**
  - `tests/test_config_equivalence.py` (6 tests: cohort static, tandem, calibrated retrain, pilot comparisons) $\rightarrow$ **PASSED**
  - `tests/test_eval_artifacts_synthetic.py` (29 tests: cache invalidation, staleness detection, manifest validation) $\rightarrow$ **PASSED**
  - `tests/test_multiview_sync_and_renderer_regression.py` (4 tests: robust bot pose, screw rejection, frame correspondence) $\rightarrow$ **PASSED**
  - `tests/test_kpms_checkpoint_compaction.py` (6 tests: compaction, continuation, numerical identity) $\rightarrow$ **PASSED**

---

## 5. Repository Layout

```text
FiberphotometryCode/
├── README.md               # Architecture documentation & quickstart
├── pyproject.toml          # Build configuration & pytest runner settings
├── Makefile                # Automation targets (test, lint, shims-report, metrics)
├── environment.lock.yml    # Reproducible conda environment definition
├── .gitignore              # Tightened rules excluding binaries, caches, outputs
├── .agents/                # Antigravity agent instructions & skills
│
├── src/fpcode/             # THE CANONICAL importable production package
│   ├── common/             # paths, schemas, h5_guard, logging, notify, run_record, io
│   ├── tracking/           # pair_matching, clock_sync, camera_correspondence, fusion
│   ├── kpms/               # init_model, checkpoints, data, runtime, fit, export, metrics, evaluation
│   ├── behavior/           # kinematics, transitions, syllables
│   ├── photometry/         # processing, metrics, alignment
│   └── viz/                # grid_movies, trajectories, figures, v2
│
├── pipelines/              # Primary CLI verbs
│   ├── train.py            # Model training & checkpoint management
│   ├── fuse.py             # Multiview posture fusion & alignment
│   ├── evaluate.py         # EML cross-evaluation & model comparison
│   ├── export.py           # Syllable extraction & h5/csv exporting
│   └── render.py           # Grid movies & trajectory generation
│
├── configs/                # Declarative YAML run configurations
│   ├── training/           # Production seeds and comparison configs
│   └── fusion/             # Static & dynamic camera fusion configs
│
├── studies/                # Modularized research studies
│   ├── behavior/           # multiview_experiments, kinematics_ulrik, replications_datta, etc.
│   └── photometry/         # turnlab, exploratory
│
├── ops/                    # Operational tooling & repo maintenance
│   ├── repo/               # metrics, triage, migrate_study_unit, quarantine_class_c
│   ├── scripts/            # batch utilities, video linkers, archives
│   ├── compaction/         # checkpoint compaction utilities
│   └── monitoring/         # training watchdogs & notification daemons
│
├── registry/               # Audit records (SHIMS.tsv, METRICS.tsv, MOVES.tsv, TRIAGE.tsv, BACKUP_DIFFS.tsv)
├── tests/                  # Fast CPU regression & equivalence test suite
├── outputs/                # Local run figures, exports, and frame dumps
├── docs/                   # Architecture specs, unit contracts, change reports
├── archive/                # Quarantined legacy backups & dead code
└── vendor/                 # Frozen forks (vendor/keypoint_moseq_syll with PATCHES.md)
```
