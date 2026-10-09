# FiberphotometryCode

Computational neuroscience platform supporting multi-camera posture tracking, unsupervised behavioral syllable segmentation (Keypoint-MoSeq), and synchronized multi-channel fiber photometry analysis. Developed in the Gether Laboratory, Department of Neuroscience, University of Copenhagen (Panum Institute).

---

## 1. Scientific & Engineering Scope

The repository provides an end-to-end analytical framework integrating high-density pose estimation with neural recording:
1. **Multi-Camera & Dual-View Tracking**: Spatial registration (affine transforms) and temporal synchronization (crystal clock pulse & hardware TTL anchors) between dorsal (top) and ventral (bottom) camera viewpoints (DeepLabCut 21/23 keypoint models).
2. **Keypoint-MoSeq (KPMS)**: Unsupervised behavioral segmentation via Autoregressive Hidden Markov Models (ARHMM) and Switching Linear Dynamical Systems (SLDS), mapping continuous kinematic trajectories into discrete postural syllables.
3. **Fiber Photometry Ingestion & Synchronization**: Multi-wavelength fluorescent biosensor recording (monoamine dynamics, calcium transients), baseline polynomial/photobleaching correction, and peri-event time histogram (PETH) alignment.
4. **Sequence & Temporal Dynamics**: State transition probability matrices, ethological sequence factorization (SeqNMF), and higher-order syntactic modeling.

> [!IMPORTANT]
> **Ethological Nomenclature Standard**: In accordance with `.agents/rules/nomenclature.md`, all automated clusters, discrete syllables, and unsupervised components are documented using **objective kinematic, spatial, and statistical descriptors** (e.g. centroid velocity, body elongation, vertical displacement, entropy) rather than subjective functional labels.

---

## 2. Repository Map

```text
FiberphotometryCode/
│
├── core/                                   # Shared foundational library modules (Single Source of Truth)
│   ├── schemas.py                          # Canonical body parts, keypoint indices & configuration models
│   ├── paths.py                            # Central workspace-relative path resolution
│   ├── camera_correspondence.py            # Affine spatial calibration & Basler crystal sync
│   ├── fusion_engine.py                    # Dual-camera dorsal-ventral keypoint fusion
│   ├── io_utils.py                         # Defensive HDF5, NumPy, and Pickle adapters
│   ├── h5_guard.py                         # Immutability enforcement for immutable checkpoints
│   ├── cache_manifest.py                   # Cryptographic sidecar manifests and provenance tracking
│   ├── eval_artifacts.py                   # Permutation-invariant EML evaluation & label permutation
│   └── logging_utils.py                    # Structured logging engine and alert dispatch
│
├── configs/                                # Versioned YAML configuration files
│   ├── tandem_cohort_50ar_200slds.yml      # Production 50-AR 200-SLDS parameters
│   └── ...
│
├── syllable_generation/                    # Primary behavioral segmentation pipeline
│   ├── multiview_top_bot_experiments/      # Canonical active production pipeline
│   │   ├── 01_data_and_fusion/             # Data ingestion, sync verification & cohort fusion
│   │   ├── 02_training_runners/            # Model fitting, seed queues & hyperparameter sweeps
│   │   ├── 03_evaluation_and_eml/          # Multi-seed EML evaluation & ranking scorecards
│   │   ├── 04_reports_and_figures/         # Trajectory renders, grid movies & diagnostic galleries
│   │   └── core/                           # Pipeline-specific engine extensions
│   ├── analysis/                           # Specialized downstream analyses
│   │   ├── replications/                   # Published replication workflows (Datta et al.)
│   │   ├── language_models/                # Markov transition orders, syntax & state entropy
│   │   ├── sequence_viz/                   # SeqNMF sequence motif factorizations
│   │   ├── shmoseq/                        # Shmoseq parameter fitting and validation
│   │   ├── syllable_chains/                # Multi-syllable transition chain dynamics
│   │   ├── ulrik_kinematics/               # Kinematic parameter extractions
│   │   └── karen_pff_vs_monomer_longitudinal_master_study/ # Longitudinal cohort study pipelines
│   └── data/                               # Session databases and preprocessed tracking arrays
│
├── fiberphotometry/                        # Fiber photometry extraction, baseline correction & PETHs
│   ├── kpms/                               # KPMS-photometry trial alignment adapters
│   ├── notebooks/                          # Interactive analysis and visualization notebooks
│   └── turnlab/                            # Turn segmentation and angular velocity routines
│
├── monoamine_interactions/                 # Multi-sensor monoamine cross-correlation studies
│   ├── notebooks/                          # Signal extraction and PETH workflows
│   └── projects/                           # Sensor baseline projects
│
├── tests/                                  # Test suite, regression benchmarks & golden datasets
│   ├── unit/                               # Fast CPU tests (math, transforms, schemas)
│   ├── integration/                        # Pipeline stage mocks & verification
│   └── golden/                             # Immutable EML benchmarks & baseline scorecards
│
├── docs/                                   # Architectural guides, operational runbooks & audits
│   ├── ARCHITECTURE.md                     # Data flow & stage pipeline design
│   ├── DATA_LAYOUT.md                      # Artifact classification (Tiers A/B/C) & storage policy
│   ├── RUNBOOK.md                          # Execution procedures, disaster recovery & resume workflows
│   └── KNOWN_ISSUES.md                     # Documented findings, anomalies & hardening ledger
│
├── scripts/                                # Maintenance, backup & archive generation utilities
│   ├── build_swe_archive.py                # Standalone code-review bundle generator
│   └── ...
│
├── ARTIFACT_INVENTORY.tsv                  # Complete repository cryptographic inventory (26,863 files)
├── DUPLICATES.tsv                          # Comprehensive hash deduplication & bundle directory audit
├── BACKUP_REPORT.md                        # Phase 0 backup verification & physical disk report
└── pyproject.toml                          # Package specification and test configuration
```

---

## 3. Quickstart & Environment

### Python Environment
Scientific computing dependencies rely on Python 3.10+, JAX (GPU/CUDA 12.4), PyTorch, DeepLabCut, and Keypoint-MoSeq:
```bash
# Recommended conda/mamba environment
conda activate kpms_gpu_cuml

# Install repository in editable mode
pip install -e .
```

### Running Verification Tests
```bash
# Run all fast CPU unit and synthetic tests
pytest -m "cpu and not needs_data"

# Run tests including data-bound fixtures (if local datasets present)
pytest -m "cpu"
```

---

## 4. Documentation References
- [System Architecture & Data Flow](file:///home/fsp585/Desktop/FiberphotometryCode/docs/ARCHITECTURE.md)
- [Data Layout & Storage Tiers](file:///home/fsp585/Desktop/FiberphotometryCode/docs/DATA_LAYOUT.md)
- [Operational Runbook & Disaster Recovery](file:///home/fsp585/Desktop/FiberphotometryCode/docs/RUNBOOK.md)
- [Known Issues & Bug Ledger](file:///home/fsp585/Desktop/FiberphotometryCode/docs/KNOWN_ISSUES.md)
- [Phase 0 Backup Report](file:///home/fsp585/Desktop/FiberphotometryCode/BACKUP_REPORT.md)

---

## 5. License
Licensed under the GNU General Public License v3.0 - see the [LICENSE](file:///home/fsp585/Desktop/FiberphotometryCode/LICENSE) file for details.