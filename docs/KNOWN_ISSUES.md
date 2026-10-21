# Known Issues & Anomaly Ledger

This document tracks known issues, historical incidents, and engineering defects identified during code audits and SWE review. In accordance with the **Scaffolding Refactoring Mandate** ("Log bugs, don't fix them; improve scaffolding only"), every item is cataloged with its scientific/computational impact and marked explicitly as **`[NOT FIXED]`**.

---

## 1. Model Checkpoint & State Incidents

### Issue 1.1: Historical In-Place Checkpoint Mutation & Seed-101 Label Swap
- **Status**: **`[NOT FIXED]`** (Documented & protected via `chmod a-w`)
- **Location**: `syllable_generation/multiview_top_bot_experiments/cohort_static_21kp_v2/`
- **Description**: An earlier EML evaluation run (2026-10-05 10:31–10:46) opened all 5 source checkpoints in `r+` mode (`fast_reindex_checkpoint`), mutating model state parameters in place. A tied-frequency state occurrence caused labels **67 $\leftrightarrow$ 68** to swap in Seed 101.
- **Impact**: Seed 101 `results.h5` (created before the run) differs from downstream outputs derived from the post-reindex checkpoint. EML ranking is unaffected due to permutation invariance, but a mapping artifact was required to reconcile legacy databases: `04_reports_and_figures/audits/seed101_old_to_new_label_map_recovered_20261005.json`.
- **Current Safeguard**: Checkpoints are now write-protected (`chmod a-w`). `core/eval_artifacts.py` enforces read-only loading and stable sorting.

---

## 2. Provenance & Manifest Gaps

### Issue 2.1: Legacy Fused H5 Caches Lack Sidecar Manifests
- **Status**: **`[NOT FIXED]`**
- **Location**: `syllable_generation/multiview_top_bot_experiments/data/fused_h5_full_cohort/` (118 sessions)
- **Description**: Fused `.h5` files created prior to October 2026 lack sidecar `.manifest.json` files storing upstream DLC hash fingerprints, camera calibration matrices, and git commit hashes.
- **Impact**: Strict cache validators report these files as `UNVERIFIED_NO_MANIFEST`. Pipelines currently default to `unmanifested_policy="trust"` to prevent massive automatic recomputation.

### Issue 2.2: Discrepancy Across Historical EML Benchmarks
- **Status**: **`[NOT FIXED]`**
- **Location**: Historical documentation and review bundles (`TANDEM_MULTIVIEW_KPMS_ENGINEERING_DOSSIER...`, `KPMS_SEED_404_EML_SELECTION...`)
- **Description**: EML rankings and likelihood tables differ slightly across different documentation summaries (e.g. Seed 404 reported as $225,001,430 \pm 234,745$ vs $224,940,120 \pm 302,124$ for Seed 101). Discrepancies stem from minor differences in frame filtering thresholds, segment orders, and AR iteration compaction.
- **Policy**: In accordance with instructions, each document's values are preserved as separate named baselines in `tests/golden/*.json` rather than picking a single synthetic winner.

---

## 3. Path & Environment Hardcoding

### Issue 3.1: Absolute Path Hardcoding (`/home/fsp585/`)
- **Status**: **`[NOT FIXED]`**
- **Location**: 1,054 Python scripts and shell files across the repository
- **Description**: Files hardcode the absolute user path `/home/fsp585/Desktop/FiberphotometryCode` or `/home/fsp585/miniforge3/...` instead of computing paths relative to the project root or reading environment variables.
- **Impact**: Prevents seamless portability to HPC compute clusters or secondary workstations.

### Issue 3.2: `sys.path` Manipulation Hacks
- **Status**: **`[NOT FIXED]`**
- **Location**: 937 Python scripts across `syllable_generation/`, `tests/`, and `fiberphotometry/`
- **Description**: Scripts rely on `sys.path.append('../..')` or `sys.path.insert(0, os.path.dirname(...))` to locate sibling modules rather than using proper packaging.
- **Impact**: Fragile import resolution when scripts are executed from outside their local directory.

### Issue 3.3: Boilerplate Code Duplication (`send_ntfy`)
- **Status**: **`[NOT FIXED]`**
- **Location**: 93 different analysis and runner scripts
- **Description**: The push notification function `send_ntfy(message)` using `urllib.request` is independently copy-pasted across dozens of files with hardcoded notification topics.
- **Impact**: Schema drift and inability to configure alerts centrally via environment variables.

---

## 4. Test Harness Anomalies

### Issue 4.1: Test Suite Polluted with Multi-Gigabyte Output Artifacts
- **Status**: **`[NOT FIXED]`**
- **Location**: `tests/runs/`, `tests/syllable_figs/`, `tests/used_frames/`
- **Description**: Over 4 GB of temporary HTML test run outputs, PNG figures, and frame dumps reside directly within the `tests/` directory tree.
- **Impact**: Bloats the test directory and confuses automated test discovery runners.

### Issue 4.2: Typo File Naming in Test Scripts
- **Status**: **`[NOT FIXED]`**
- **Location**: `tests/teest_oft_session.py`, `tests/teest_simple_session.py`
- **Description**: Typo prefix (`teest_`) prevents standard pytest auto-discovery from running these tests.
- **Impact**: Tests are skipped during standard `pytest` runs unless invoked explicitly.

### Issue 4.3: Hardcoded Test Constants (`test_multiview_north_stars_4_1_to_4_3.py`)
- **Status**: **`[NOT FIXED]`**
- **Location**: `tests/test_multiview_north_stars_4_1_to_4_3.py`
- **Description**: Test asserts hardcoded legacy 118-session EML numbers and reads legacy relative rankings paths without fallback.
- **Impact**: Fails if legacy rankings files are moved or if data is updated. Preserved in `tests/golden/eml_baseline_118sess_legacy_tandem.json`.

### Issue 4.4: Signature Drift in `tests/test_flipping.py`
- **Status**: **`[NOT FIXED]`**
- **Location**: `tests/test_flipping.py`
- **Description**: Calls `flip_data(data, conf_dict, bodyparts, mode='augmentation')` without the `image_width` parameter. The canonical `flip_data` function was updated to require `image_width` (correctly validated in `tests/test_flipping_width.py`).
- **Impact**: Fails with `TypeError: flip_data() missing 1 required positional argument: 'image_width'`.

### Issue 4.5: AST Pattern Mismatch in `tests/test_v4_repairs.py`
- **Status**: **`[NOT FIXED]`**
- **Location**: `tests/test_v4_repairs.py:123`, `tests/test_v4_repairs.py:136`
- **Description**: AST inspections search for `render_superimposed` and `for b in manifest_bouts:` within `render_all_cohort_static_21kp_v2_seed101_trajectories.py`, which was refactored during visualization repairs.
- **Impact**: Raises `KeyError` and `StopIteration` during AST traversal.

### Issue 4.6: Module-Level Heavy Execution in `tests/test_oft_session_full_data.py`
- **Status**: **`[NOT FIXED]`**
- **Location**: `tests/test_oft_session_full_data.py:52`
- **Description**: Exported Jupyter notebook executes `sessions = load_all_sessions(path, ...)` at global module import time rather than deferred inside a test function.
- **Impact**: Ingests multi-gigabyte OFT datasets into RAM during test collection. Ignored in `pytest.ini` (`--ignore=tests/test_oft_session_full_data.py`).
