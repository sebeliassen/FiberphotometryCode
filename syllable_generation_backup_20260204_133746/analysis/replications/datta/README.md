# Datta: Modular Fiber Photometry Correlation Pipeline

This library is a modular refactor of the legacy Datta-style correlation analysis. It maintains **bit-for-bit parity** with the original logic while providing a cleaner, more maintainable, and highly documented codebase.

## Directory Structure
- `lib/`: Core library modules.
  - `config.py`: Centralized configuration (Bin scales, shuffle counts, paths).
  - `features.py`: Kinematics (acceleration, jerk, duration) and metadata extraction.
  - `stats.py`: Binning, Z-scoring, and Pearson correlation engine.
  - `shuffling.py`: Permutation and circular roll shuffles (Parallelized).
- `run_pipeline.py`: Main driver to process sessions and generate raw Parquets.
- `digest_results.py`: Aggregates raw Parquets into statistical Pickles (Chance bands, P-norms).
- `plot_results.py`: Generates final publication-quality plots with Holm-corrected significance bars.
- `output/`: Results, including `plots/` and cached statistical `*.pkl` files.
- `verification/`: Tools used to prove parity against the legacy pipeline (Synthetic "Twin" pipeline).

## Usage Flow
1. **Configure**: Update `lib/config.py` (e.g., set `DEBUG_MODE = False`, `NSHUFF = 300`).
2. **Execute**:
   ```bash
   python run_pipeline.py   # Step 1: Generate Raw Data
   python digest_results.py # Step 2: Digest Stats
   python plot_results.py   # Step 3: Visualise
   ```

The Datta pipeline has been validated using a **Twin Pipeline** approach on synthetic data:
- **Numerical Parity**: `MAE = 0.0000` for all observed and shuffle correlations.
- **Structural Parity**: CSV-level comparison of plot traces, bands, and significance flags showed **0.00 difference**.
- **Legacy Bug Handling**: Correctly replicates known legacy behaviors (e.g., categorical mean aggregation) while providing fixes for visual issues (e.g., improved plot band visibility).

## For Future AI Agents
- **Metadata**: The pipeline relies on `session_id`, `animal_id`, `genotype`, and `signal_meta` (extracted from the session objects).
- **Parallelism**: Uses `joblib` for shuffles. Configure `NJOBS` in `config.py`.
- **Plotting**: Holm-Bonferroni correction is applied per Brain Region across all features and timescales.
