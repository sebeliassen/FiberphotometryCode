# Test Suite Audit & Reorganization Plan

## 1. Current State Audit of `tests/`

The current `tests/` directory spans **4.0 GB** and conflates automated test code with test execution artifacts and interactive notebooks:

| Item | Type | Size | Primary Issue |
|---|---|---|---|
| `tests/runs/` | Directory | **3.9 GB** | 1,400+ HTML test reports (`out_idx*.html`), JSON dumps, and exported notebooks generated during testing. |
| `tests/syllable_figs/` | Directory | ~45 MB | Diagnostic PNG figures output during test execution. |
| `tests/used_frames/` | Directory | ~15 MB | Extracted video frame dumps. |
| `tests/teest_oft_session.py` | Python Script | 6.8 KB | Typo in filename prefix (`teest_`) prevents standard pytest auto-discovery. |
| `tests/teest_simple_session.py` | Python Script | 4.2 KB | Typo in filename prefix (`teest_`) prevents standard pytest auto-discovery. |
| `tests/*.ipynb` | Notebooks | ~25 MB | 10 interactive test exploration notebooks sitting in test root. |
| `tests/test_*.py` | Python Modules | ~350 KB | Active automated unit and synthetic regression test modules. |

---

## 2. Target Test Architecture

The refactored test structure cleanly decouples automated assertions from outputs, categorizes tests by speed and hardware requirements, and integrates golden reference benchmarks:

```text
tests/
├── conftest.py                     # Global pytest fixtures + require_artifact() skipper
├── pytest.ini                      # Marker registrations (cpu, gpu, slow, needs_data)
│
├── unit/                           # Fast CPU-only mathematical and algorithmic tests (< 1 sec)
│   ├── test_schemas.py             # SSOT keypoint and bodypart invariant tests
│   ├── test_camera_correspondence.py # Affine registration matrix math
│   ├── test_flipping_invariance.py # Coordinate inversion symmetry
│   └── test_velocity_subsetting.py # Velocity filtering thresholds
│
├── integration/                    # Pipeline stage mocks & multi-camera synthetic tests
│   ├── test_cache_manifest_synthetic.py # Sidecar provenance and hash verification
│   ├── test_fusion_provenance_and_cache_synthetic.py # Dual-camera synchronization
│   ├── test_eval_artifacts_synthetic.py # Permutation-invariant EML loader
│   └── test_kpms_checkpoint_compaction.py # Rolling storage compaction logic
│
├── golden/                         # Versioned immutable benchmark scorecards
│   ├── eml_baseline_118sess_legacy_tandem.json
│   ├── eml_baseline_continuous_calibrated_hardening.json
│   └── multiview_sync_and_fusion_rules.json
│
└── notebooks/                      # Exploratory test notebooks (decoupled from CI runners)
    ├── test_oft_session_full_data.ipynb
    └── ...
```

---

## 3. Output Redirection Strategy (Phase 3 Planned Moves)

To prevent repository bloat, output generation will be redirected outside `tests/`:
- `tests/runs/` $\rightarrow$ `outputs/test_runs/` (leaving symlink at `tests/runs/` for legacy compatibility).
- `tests/syllable_figs/` $\rightarrow$ `outputs/test_figures/` (leaving symlink at `tests/syllable_figs/`).
- `tests/used_frames/` $\rightarrow$ `outputs/test_frames/` (leaving symlink at `tests/used_frames/`).

---

## 4. Typo Rectification Plan (Phase 3 Planned Moves)

- `tests/teest_oft_session.py` $\rightarrow$ `tests/test_oft_session.py` (with compatibility shim at `teest_oft_session.py`).
- `tests/teest_simple_session.py` $\rightarrow$ `tests/test_simple_session.py` (with compatibility shim at `teest_simple_session.py`).

---

## 5. Execution Markers & CI Conventions

```bash
# Run all fast unit tests (standard local development & pre-commit)
pytest -m "cpu and not slow and not needs_data"

# Run integration suite
pytest tests/integration/ -m "cpu"

# Run tests that verify against golden EML baselines
pytest -m "cpu" tests/test_multiview_north_stars_4_1_to_4_3.py
```
