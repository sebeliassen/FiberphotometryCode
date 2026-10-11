# Replication Unification: Project-Aware Architecture

**Context**: To minimize code duplication and maintenance overhead, the Datta replication pipeline (specifically Panel 1f) was refactored into a unified, project-aware architecture. This allows a single suite of scripts to handle both IFDN and AMPH datasets by parameterizing project-specific logic (e.g., file paths, drug metadata).

## Core Philosophy
1.  **Single Source of Truth**: Analytical logic (digestion, statistics, plotting) resides in one place to ensure consistency across datasets.
2.  **CLI Parameterization**: Use `--project {ifdn,amph}` to toggle context-specific behavior.
3.  **Strict Parity**: The unified pipeline was validated against legacy project-specific scripts to ensure 100% numerical and visual identity.

## Unified Components

### 1. Data Contexts
The pipeline automatically handles the primary differences between Datta datasets:
- **IFDN**: Standard baseline/genotype comparisons.
- **AMPH**: Dynamically includes `drug` (Saline/Amph) metadata in groupings and plotting.

### 2. Analytical Drivers
- **`run_panel_1f_unified.py`**: Extraction engine. Uses sharded Parquet output for parallel efficiency. 
- **`digest_panel_1f_unified.py`**: The "brain" of the pipeline. Performs aggregation and shuffle-based normalization.
    - *Improvement*: **Redundancy Suppression**. Automatically detects if only one hemisphere exists for a group and suppresses the "Avg" output to prevent duplicate plots.
- **`plot_panel_1f_unified.py`**: The visualizer. Produces the specific dopamine transient traces.

## Usage Pattern

The workflow follows a standard sequence:
```bash
# 1. Extraction (Optional if shards exist)
python run_panel_1f_unified.py --project ifdn --n-jobs 8

# 2. Digestion 
# Supports mode: std, midfi, hifi
python digest_panel_1f_unified.py --project ifdn --mode midfi --shuffle-limit 100

# 3. Plotting
python plot_panel_1f_unified.py --project ifdn --suffix _merged
```

## Legacy Archival
With the successful verification of this architecture, project-specific legacy scripts (e.g., `digest_panel_1f_ifdn.py`) were archived or deleted, and verification parity checkers were offloaded to `syllable_generation/diagnostics/`.
