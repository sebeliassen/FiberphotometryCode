
# Panel 1h Trace Mode Replication

## Overview

This module replicates the "Trace" (Time-Series Heatmap) analysis from the datta lab's `1h_right_all.ipynb` notebook. It was developed to achieve **bit-exact mathematical parity** with the original implementation while integrating into the unified pipeline structure.

## Key Features

1.  **Dual Mode Support**: Supports both `peak` (correlation analysis) and `trace` (heatmap analysis) modes via the `--type` argument.
2.  **Drug Splitting**: Correctly handles dataset-specific splitting:
    -   **IFDN**: Single "baseline" condition.
    -   **AMPH**: Splits into "Saline" and "Amphetamine" conditions.
3.  **Parity Verified**: The implementation has been verified to produce identical numerical outputs (Z-scores) to the legacy notebook logic.
    -   Max Difference: 0.0 for both IFDN and AMPH datasets.

## Implementation Details

### Digestion (`digest_panel_1h_unified.py`)
-   **Trace Digestion**:
    -   Applies 1st-order Bessel High-Pass Filter (1Hz).
    -   Extracts syllable-centered windows (+/- 1.5s).
    -   Computes actual mean response vs. 20x shuffled mean response (jittered onsets).
    -   Saves results as nested dictionaries in `.pkl` files.

### Plotting (`plot_panel_1h_unified.py`)
-   **Trace Plotting**:
    -   Calculates Z-scores using the global shuffle mean and standard deviation: `(actual - shuffle_mean) / shuffle_std`.
    -   Applies row-wise normalization and Gaussian smoothing (sigma=1.0).
    -   Sorts rows by peak activity in the post-onset window.
    -   Generates separate plots for each Drug, Hemisphere, and Fiber Color.

## Usage

Run the unified driver script:

```bash
# Run Trace analysis for IFDN project
python run_panel_1h_unified.py --project ifdn --type trace

# Run Trace analysis for AMPH project
python run_panel_1h_unified.py --project amph --type trace

# Run BOTH Peak and Trace analyses sequentially
python run_panel_1h_unified.py --project amph --type all
```

## Comparisons

Side-by-side comparisons of the legacy notebook output vs. the unified pipeline output are stored in `plots/side_by_side/`.
