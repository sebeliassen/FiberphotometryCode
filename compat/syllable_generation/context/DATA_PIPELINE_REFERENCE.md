# Data Pipeline Reference

This document details the end-to-end pipeline for processing fiber photometry and behavioral data, from raw ingestion to the final analysis-ready artifacts.

## Overview
The pipeline follows a **Load → Enrich → Attach** pattern:
1.  **LOAD**: Raw data is ingested into "Dirty" session objects.
2.  **ENRICH**: Metadata (Drug, Dose, Genotype) is injected, and Exclusions are applied to create "Sanitized" objects.
3.  **ATTACH**: Model results (Syllables, Latent States) are attached and synchronized to the 80Hz photometry timeline.

## 1. Loader & Enrichment Phase
**Goal**: Create a clean, metadata-rich session pickle (`_sanitized.pkl`).

### A. Raw Loading
*   **Script**: `loaders/create_jonathan_sessions.py` (IFDN) or `create_jonathan_amph.py` (AMPH).
*   **Process**:
    *   Reads TDT/CSV files.
    *   Aligns Photometry channels.
    *   Detects "Entry Time" (optional).
    *   **Output**: A raw "Dirty" pickle (in-memory or saved as `_dirty.pkl`).

### B. Enrichment (Dirty → Sanitized)
*   **Module**: `projects/PROJECT_NAME/enrich.py`
*   **Logic**:
    *   **Metadata Injection**: Reads `animals.csv` and `sessions.csv` to inject `AnimalID`, `Genotype`, `Drug`, `Dose`.
    *   **Exclusion Application**: Reads `exclusions.csv` to mask invalid data (e.g., fiber detachments).
        *   **Manual Exclusions**: User-defined time ranges (set to NaN).
        *   **Auto Masks**: Automated signal quality checks (set to NaN).
    *   **Flag**: `skip_exclusions=True` produces "Dirty" data; `False` produces "Sanitized".
*   **Output**: `jonathan_PROJECT_sessions_sanitized.pkl` (Saved to `syllable_generation/data/`).

## 2. Syllable Attachment Phase
**Goal**: Merge behavioral model results with the sanitized session data.

### A. The Core Logic (`attach_syllables_kpms.py`)
This script is the unified driver for both IFDN and AMPH datasets.

**Command**:
```bash
python3 analysis/attach_syllables_kpms.py \
    --input data/jonathan_ifdn_sessions_sanitized.pkl \
    --output data/jonathan_ifdn_sessions_w_sylls.pkl \
    --kpms path/to/results.h5
```

### B. Synchronization & Upsampling
The script automatically handles the sampling rate mismatch between the Model (e.g., 40Hz) and Photometry (e.g., 80Hz).

1.  **Load VFI**: Loads `video_frame_indexes.npz` (VFI) which maps every Model Step -> Video Frame.
2.  **Auto-Rate Detection**:
    *   Calculates `Ratio = Session_Length / VFI_Max`.
    *   ** IFDN**: Ratio ≈ 1.0 (Native 80Hz).
    *   ** AMPH**: Ratio ≈ 2.0 (Model 40Hz -> Photometry 80Hz).
3.  **VFI Scaling**: If Rate > 1.0, the VFI indices are scaled (multiplied by Rate) to match the Photometry timeline.
4.  **Syllable Upsampling**:
    *   If using a pre-upsampled VFI (rare/advanced), syllables are upsampled using Nearest Neighbor.
    *   Otherwise, standard VFI scaling handles the alignment.
5.  **Flattening**: Behavioral results are flattened onto the Photometry timeline using the VFI.

## 3. Final Artifact Structure
The output `_w_sylls.pkl` contains a list of `Session` objects with:

*   **`s.metadata`**: Rich dictionary containing all categorical info (Drug, Genotype, Exclusions).
*   **`s.all_data_df`**: A "Slim" DataFrame aligned to 80Hz.
    *   **Columns**: `SystemTimestamp`, `signal_*` (Photometry), `centroid_*` (Kinematics), `syllable`, `frame_index`.
    *   **NO Redundancy**: Metadata is NOT duplicated as columns here.

## 4. Replication Steps

To regenerate the entire dataset from scratch:

1.  **Generate Sanitized Inputs**:
    ```bash
    python3 loaders/create_jonathan_sessions.py  # Generates IFDN sanitized
    python3 loaders/create_jonathan_amph.py      # Generates AMPH sanitized
    ```

2.  **Attach Syllables (IFDN)**:
    ```bash
    python3 analysis/attach_syllables_kpms.py \
        --input data/jonathan_ifdn_sessions_sanitized.pkl \
        --output data/jonathan_ifdn_sessions_w_sylls.pkl \
        --kpms path/to/ifdn/checkpoint.h5
    ```

3.  **Attach Syllables (AMPH)**:
    ```bash
    python3 analysis/attach_syllables_kpms.py \
        --input data/jonathan_amph_sessions_sanitized.pkl \
        --output data/jonathan_amph_sessions_w_sylls_final.pkl \
        --kpms path/to/amph/checkpoint.h5
    ```

4.  **Verify**:
    ```bash
    ./diagnostics/check_sampling_rates.py
    ```

## 5. Analysis Replication (Datta)
**Goal**: Replicate and extend the legacy "Datta" analysis (from `datta_orig`) in a modular, maintainable, and high-performance pipeline.

### Location
`syllable_generation/analysis/replications/datta/`

### A. Panel 1e: Scalar Correlations
Calculates the correlation between behavioral scalars (Velocity, Acceleration) and Neural Signals across timescales (1s to 60s).

#### 1. Pipeline Structure (`datta/lib/`)
*   **`config.py`**: Centralized constants (BINS, SHUFFLES).
*   **`features.py`**: Kinematic extraction (Acc, Jerk) and Signal Metadata parsing.
*   **`stats.py`**: Core binning and correlation logic.
*   **`shuffling.py`**: Parallelized shuffle workers (Roll/Permute).

#### 2. Drivers
*   **IFDN (Baseline)**: `run_pipeline.py`
    *   Replicates original analysis bit-for-bit.
    *   Output: `output/panel_1e/`
*   **AMPH (Expansion)**: `run_panel_1e_amph.py`
    *   Adds **Drug** metadata handling (Saline vs Amphetamine).
    *   Uses identical parallel optimizations.
    *   Output: `output/panel_1e_amph/`

#### 3. Visualization
*   **Generate Plots**: `plot_results.py` (IFDN) or `plot_panel_1e_amph.py` (AMPH).
*   **Features**: Includes 95% Chance Bands (Gray) and Significance Markers (Holm-corrected).

### B. Panel 1f: Event-Triggered Snippets
(In Progress) Aggregates neural traces aligned to specific behavioral events.

*   **Events**: Syllable Onsets, Movement Initiations.
*   **Drivers**:
    *   `run_panel_1f.py`: Event detection and snippet extraction.
    *   `plot_panel_1f.py`: Visualizes traces (Mean ± 95% CI) per Region/Genotype.
