# Software Engineering & Architecture Review Prompt

You are acting as a Principal Research Software Engineer and Python Architect specializing in scientific computing, machine learning pipelines, and computational neuroscience.

You are provided with a complete code review archive (`fiberphotometry_code_review.zip`) of the `FiberphotometryCode` repository.

---

## 1. Repository Background & Scientific Domain
This codebase supports advanced behavioral neuroscience and fiber photometry analysis:
- **Keypoint-MoSeq (KPMS)**: Unsupervised behavioral segmentation of animal postures into discrete syllables using Autoregressive Hidden Markov Models (ARHMM) and SLDS.
- **Multi-Camera & Tandem View**: Synchronized top and bottom camera tracking (DeepLabCut 21/23 keypoints), affine camera registration, pseudo-3D posture modeling, and behavioral syllable classification.
- **Fiber Photometry Analysis**: Synchronized calcium / monoamine sensor signal processing (e.g. DA, 5-HT, NE), baseline correction, peri-event time histograms (PETHs), and behavioral correlation.
- **Sequence Analysis**: Ethological sequence analysis (SeqNMF, syllable chains, transition matrices).

---

## 2. Archive Construction & Placeholder Convention
To allow full repository review without gigabytes of raw video, DLC tracking data, and model checkpoints:
- **Source Code**: Python files (`.py`), Markdown (`.md`), Shell scripts (`.sh`), and configuration files (`.yaml`, `.json`, etc.) are included **verbatim** in full.
- **Jupyter Notebooks**: All `.ipynb` notebooks are included with execution cell outputs stripped (retaining all markdown and Python code cells).
- **Data & Binaries as Placeholders**: All large non-selfwritten files (CSVs, HDF5 `.h5`, Pickles `.pkl`, video recordings `.avi`/`.mp4`, plots `.png`, and zip backups) are replaced with **`<original_filename>.txt`** placeholder files. Each placeholder records:
  - Original filename and relative path
  - Detected file type / format
  - Exact byte size and human-readable size
  - Last modified timestamp
  - Symlink details (if applicable)

### Summary Statistics of this Archive:
- **Total Files Represented**: 26,860
- **Verbatim Code & Config Files**: 4,630
- **Stripped Jupyter Notebooks**: 121
- **Lightweight Placeholders**: 22,109 (representing 3138.38 GB of raw data/binaries)

---

## 3. Your Review Objectives & Expected Deliverables

Please conduct a thorough, rigorous software engineering audit of this repository and provide an actionable, prioritized roadmap across the following 6 core pillars:

### Pillar 1: Repository Structure & Directory Hierarchy
- Identify organizational anti-patterns (e.g., redundant backup folders like `syllable_generation_backup_*`, `multiview_top_bot_experiments_code_backup_*`, root-level clutter, orphaned scratch folders).
- Propose a clean, standardized monorepo or multi-package layout following modern Python standards (e.g., `src/`, `configs/`, `pipelines/`, `scripts/`, `docs/`, `tests/`).
- Recommend proper segregation between **raw input data**, **cached intermediate features**, **model checkpoints**, and **final presentation figures/deliverables**.

### Pillar 2: Architectural Modularity & Separation of Concerns
- Analyze script vs library boundaries: many scripts mix CLI arguments, data loading, heavy mathematical fitting, visualization generation, and file I/O in single monolithic files.
- Design reusable module boundaries (e.g., `io`, `calibration`, `modeling`, `metrics`, `visualization`).
- Address hardcoded assumptions (e.g., absolute filesystem paths `/home/fsp585/...` vs configuration management using Hydra, Pydantic, or YAML).

### Pillar 3: Code Duplication & Dead Code Pruning
- Identify cloned or slightly diverged duplicate scripts across different experiment directories.
- Identify legacy prototypes that should be archived or merged into canonical runners.
- Highlight common utility functions that should be consolidated into a shared core package.

### Pillar 4: Reproducibility, Configuration & Environment Management
- Audit dependencies, requirements, and environment management (e.g., `pyproject.toml`, conda `environment.yml`).
- Recommend best practices for hardware-accelerated scientific computing (GPU/CUDA memory scaling, page cache compaction, PyTorch/JAX memory management).
- Propose data artifact tracking (e.g., DVC or lightweight manifest hashing) so experiments can reference inputs/checkpoints cleanly.

### Pillar 5: Testing, Validation & Continuous Integration
- Review the current `tests/` directory (noting generated test run outputs vs actual unit/integration tests).
- Design a test strategy including:
  - Fast unit tests for geometry/affine math and photometry signal processing.
  - Mocked integration tests for KPMS model fitting and pipeline runners.
  - Regression tests for syllable classification and trajectory generation.
- Suggest a lightweight GitHub Actions / CI pipeline.

### Pillar 6: Prioritized Step-by-Step Refactoring Roadmap
Provide a phased implementation plan:
- **Phase 1: Immediate Hygiene & Quick Wins** (safe file moves, `.gitignore` hardening, root cleanup, removing dead backups).
- **Phase 2: Core Package Consolidation** (establishing canonical packages, shared utilities, and config schema).
- **Phase 3: Pipeline & CLI Standardization** (refactoring runners to use unified CLI/config interfaces).
- **Phase 4: CI/CD & Testing Infrastructure**.

---
Please begin your audit by analyzing the top-level repository structure and identifying the highest-impact architectural improvements.
