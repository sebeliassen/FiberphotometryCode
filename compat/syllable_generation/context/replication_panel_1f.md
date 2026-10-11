# Panel 1f Replication: Movement vs Syllable Initiation Traces

**Context**: This documentation details the logic and implementation for replicating "Panel 1f" from the Datta Lab's methodology (e.g., Markowitz et al., 2023). The goal is to visualize dopamine (DA) transients aligned to the onset of spontaneous syllable usage versus movement initiations.

## Pipeline Modes (Fidelity Tiers)

To ensure both visual clarity and statistical rigor, the pipeline supports three processing modes controlled via a command-line argument to `run_replication_pipeline.sh`:

### 1. Mid-Fi (Selected Standard)
*   **Command**: `./run_replication_pipeline.sh midfi`
*   **Logic**:
    *   **Filtering**: None. Traces are kept "raw" (no 1Hz hipass) to preserve signal smoothness and avoid high-frequency noise artifacts ("scribbles").
    *   **Normalization**: **Z-to-Shuffle**. The observed mean trace is renormalized against the shuffle distribution:
        $$Z_{shuffle} = \frac{\mu_{obs} - \mu_{shuffle}}{\sigma_{shuffle}}$$
    *   **Units**: "DA (Shuffle STDs)". This aligns the Y-axis magnitude with published figures.
*   **Output**: `plots_midfi/`

### 2. Hi-Fi ("Pixel Perfect")
*   **Command**: `./run_replication_pipeline.sh hifi`
*   **Logic**:
    *   **Filtering**: **1Hz Bessel High-Pass Filter** (2nd order). Strictly follows the Datta lab methods to remove baseline drift.
    *   **Normalization**: Z-to-Shuffle.
*   **Trade-off**: While methdologically identical to the original paper, on this specific dataset, the high-pass filter exacerbated the appearance of the noise floor, creating distracting "scribbly" traces.
*   **Output**: `plots_hifi/`

### 3. Standard
*   **Command**: `./run_replication_pipeline.sh std`
*   **Logic**:
    *   **Filtering**: None.
    *   **Normalization**: Standard Z-Score (dF/F).
*   **Output**: `plots_std/` (or default `plots/`)

## Glossary of Terms

### Alignment Events
*   **`syllables_offline`**: Syllable onset timestamps derived from the **full, bidirectional** VFI (Video Frame Index) analysis. Because the model analyzes the entire session file at once, it can "see the future," providing the most accurate ground truth for when a behavioral syllable actually began. This is preferred over "online" (real-time) detection for alignment.
*   **`movement`**: A control metric. These are initiation indices calculated based on the velocity derivative crossing a threshold (Markowitz 2022 logic).

### Processing Steps
*   **Aggregated Region/Hemi**: Results are split by Genotype (WT/KI) -> Drug (Saline/Amph) -> Brain Region (DMS/DLS/VS) -> Hemisphere (Left/Right/average).
*   **Green Fiber Focus**: Only signals with `orig_letter='G'` (Green/GCaMP) are processed.
*   **Flat Shuffle Bands**: The 95% Confidence Interval for the shuffle distribution is visualized as a static horizontal band (mean of 2.5% and 97.5% percentiles across the window) rather than a time-varying envelope, improving legibility.
