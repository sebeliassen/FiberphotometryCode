# Data Layout & Storage Tiers

This document details the classification, lifecycle rules, regenerability criteria, and backup policies for all data artifacts in the `FiberphotometryCode` repository (~3.1 TB total footprint).

---

## 1. Storage Tiers Overview

Every file in the repository belongs to one of three primary data tiers or to code/documentation:

| Tier | Category | Total Volume | Regenerable? | Backup Policy | Retention Rule |
|---|---|---|---|---|---|
| **Tier A** | Checkpoints, `results.h5`, `pca.p`, init models, session pickles, configs, manifests | **905.39 GB** (576 files) | **NO** (Irreplaceable model states) | Physical redundancy on `/mnt/hdd/backups/TierA_backup_20261009/` (<2GB); large files write-protected in place | Permanent; write-protected via `chmod a-w` |
| **Tier B** | Fused H5s, preprocessed feature caches, intermediate diagnostics, test run outputs | **197.36 GB** (16,461 files) | **YES** (Deterministically reproducible via pipelines) | Cryptographically indexed in `ARTIFACT_INVENTORY.tsv`; copied to backup disk only if $\ge$20% free space remains | Scratch/cache lifecycle; can be regenerated from raw data |
| **Tier C** | Raw DLC H5s, tracking CSVs, camera video recordings (`.avi`, `.mp4`), raw photometry | **268.61 GB** on SSD (+ ~2.5 TB on external HDD) | **NO** (Primary empirical data) | Read-only input source; indexed in `ARTIFACT_INVENTORY.tsv` | Permanent read-only archive; never modified by pipelines |
| **CODE / DOCS** | Python source, Jupyter notebooks, shell scripts, configs, markdown documents | **0.21 GB** (2,346 files) | **YES** (Git versioned) | Tracked in Git branch `scaffolding-refactor` (tag `pre-scaffolding-20261009`) + `code_snapshot.tar.gz` | Version-controlled permanently |

---

## 2. Tier Details & File Conventions

### Tier A: Critical & Irreplaceable Scientific Artifacts

These files represent the mathematical convergence of heavy GPU model fitting. Because training runs require days of compute and cannot be trivially rerun, they are protected against modification:

- **Model Checkpoints (`checkpoint.h5`, `checkpoint.pt`)**:
  - Found in: `syllable_generation/multiview_top_bot_experiments/**/checkpoint.h5`, `kpms_projects/**/checkpoint.h5`
  - Footprint: Individual files range from 2.2 GB up to 113.17 GB (`cohort_static_21kp_v2/model_21kp_static_50ar_300slds_seed101/checkpoint.h5`).
  - Protection: Write permissions removed (`chmod a-w`). Immutability enforced by `core/h5_guard.py`.
- **Principal Components (`pca.p`, `pca.pkl`)**:
  - Contains posture decomposition eigenvalues and spatial mean centering vectors.
  - Required for projecting any future tracking data into the identical latent space.
- **Initialization States (`init_model_seed_*.pkl`)**:
  - ARHMM model priors and initial transition matrices for each random seed (Seeds 101, 202, 303, 404, 505).
- **Session Databases (`sessions_*.pkl`)**:
  - Cleaned session dictionaries binding behavioral syllables, video metadata, and photometry signals.
- **Manifests & Configs (`*.yml`, `*.yaml`, `*manifest*.json`)**:
  - Machine-readable specifications defining experimental groupings, parameter overrides, and tracking provenance.

### Tier B: Regenerable Intermediate Data & Caches

These artifacts are computationally derived from Tier C inputs and can be rebuilt if storage reclamation is needed:

- **Dual-Camera Fused Datasets (`*fused*.h5`, `*reflection_corrected*.h5`)**:
  - Output of `01_data_and_fusion/run_cohort_fusion.py`.
  - Reconstructible from raw top and bottom DLC H5 files using the static affine calibration matrix.
- **Precomputed Feature Caches (`preprocessed_data.pkl`, `*.cache`)**:
  - Coordinate arrays after confidence filtering, median filtering, and recentering.
- **Visualizations & Diagnostics (`visualizations_*/`, `tests/runs/`, `grid_movies*/`)**:
  - Downstream diagnostic renderings, HTML galleries, and test output bundles.
  - Automatically regenerable via `04_reports_and_figures/`.

### Tier C: Empirical Raw Datasets (Immutable Inputs)

- **Raw DeepLabCut Tracking Arrays**:
  - Dorsal (TopCam) and Ventral (BotCam) 2D tracking predictions with likelihood confidence scores.
- **High-Speed Video Recordings (`.avi`, `.mp4`)**:
  - Dual-camera recordings stored across primary SSD and secondary high-capacity storage (`/mnt/hdd/`).
- **Photometry Data Dumps**:
  - Multi-fiber raw fluorescence traces, excitation wavelengths (405nm, 470nm, 560nm), and timestamp logs.

---

## 3. Directory Layout by Data Tier

```text
FiberphotometryCode/
├── configs/                                 <- [Tier A] Version-controlled YAML specs
├── ARTIFACT_INVENTORY.tsv                   <- [Tier A] Cryptographic manifest of all 26,863 files
├── syllable_generation/
│   ├── multiview_top_bot_experiments/
│   │   ├── configs/                         <- [Tier A] Production YAML configs
│   │   ├── cohort_static_21kp_v2/
│   │   │   └── **/checkpoint.h5            <- [Tier A] 113 GB production model checkpoint (read-only)
│   │   ├── kpms_tandem_cohort_continuous_calibrated/
│   │   │   ├── **/checkpoint.h5            <- [Tier A] 31 GB seed checkpoints (read-only)
│   │   │   └── init_model_seed_*.pkl        <- [Tier A] 15.6 GB initialization models (read-only)
│   │   ├── data/
│   │   │   ├── fused_h5_full_cohort/        <- [Tier B] 118 dual-camera fused H5 files (6.3 GB)
│   │   │   └── session_caches/              <- [Tier A] Production session pickles (read-only)
│   │   └── 04_reports_and_figures/
│   │       └── visualizations_*/            <- [Tier B] Syllable manifests & HTML galleries
│   └── data/                                <- [Tier A] Baseline session databases
├── tests/
│   ├── golden/                              <- [Tier A] Immutable EML reference scorecards
│   └── runs/                                <- [Tier B] Test execution reports and rendered artifacts
└── /mnt/hdd/backups/TierA_backup_20261009/  <- [Backup] Off-disk physical backup of Tier A
```
