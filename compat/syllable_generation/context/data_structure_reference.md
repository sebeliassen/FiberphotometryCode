# Session Data Structure Reference

## Overview
The pipeline operates on `Session` objects loaded from pickle files (`jonathan_{project}_sessions_*.pkl`). Each session represents one recording from one animal, containing synchronized behavioral (MoSeq) and neural (Photometry) data.

## 1. The Session Object
Each `Session` object `s` has two primary attributes:
*   `s.metadata` (Dict): Configuration and experimental details.
*   `s.all_data_df` (DataFrame): The main time-series data.

## 2. Main Dataframe (`s.all_data_df`)
This dataframe is aligned to the native video recording rate (80Hz) and synchronized with fiber photometry (20Hz base, or 80Hz interleaved multiplex).

### Time & Identification
*   `frame_index` (int): synchronized frame number.
*   `SystemTimestamp` (float): absolute time.
*   `epoch_time`: (optional) Unix timestamp.

### Behavior (Kinematics)
*   `centroid_x`, `centroid_y` (float): Animal position (pixels).
*   `velocity_px_s` (float): Speed (pixels/second).
*   `angular_velocity` (float): Turning speed.
*   `heading` (float): Head direction (radians).

### Behavior (MoSeq)
*   `syllable` (int): The behavioral syllable ID assigned by Keypoint-MoSeq.
*   `onset` (bool): True at the start of a new syllable.

### Neural Signals (Photometry)
*   `signal_0` ... `signal_3` (float): Raw fluorescent signal.
*   `signal_0_dff` (float): $\Delta F/F$ calculated via isosbestic subtraction.
*   `signal_0_zscore` (float): Z-scored $\Delta F/F$ (full session mean/std).

---

## 3. Project Differences (IFDN vs AMPH)

### IFDN (Instrumental Focus / Genetic)
*   **Context**: Genetic mutation (e.g., IFDN1 KO) vs Wild Type.
*   **Structure**: Continuous recording (approx 20-30 mins).
*   **Key Metadata**:
    *   `genotype`: "WT" or "HET"/"HOM" (or "KI").
    *   `group`: Derived grouping for analysis.

### AMPH (Amphetamine Challenge)
*   **Context**: Saline vs Amphetamine injection challenge.
*   **Structure**: Pre-injection baseline $\to$ **Injection Event** $\to$ Post-injection response.
*   **Key Metadata**:
    *   `drug`: "Saline" or "Amphetamine".
    *   `dose_mg_kg`: Dosage.
    *   `injection_frame`: Frame index where injection occurred.
    *   `entry_frame`: Frame index where mouse re-entered the arena.
*   **Analysis Logic**:
    *   **Filtering**: Analysis modules often truncate the data to `[entry_frame + delay : ]` to analyze the drug effect window.
    *   **Splitting**: Some comparisons measure "Pre vs Post", while most current analyses focus on "Saline Post vs Amph Post".

---

## 4. Metadata Reference (`s.metadata`)
Common keys found in `s.metadata`:
*   `animal_id`: Unique identifier (e.g., '69536').
*   `fs`: Sampling rate (e.g., 30.0 Hz).
*   `photometry_channels`: Mapping of signals (415nm, 470nm, etc.).
*   `fiber_indices_auto`: Helper for multi-fiber mapping.
*   `signal_keys`: Dictionary mapping brain regions to signal columns (e.g., `{'L-DLS': 'signal_0_zscore'}`).

## 5. How It's Used in Analysis
1.  **Loading**: `loader.load_sessions(project)` reads the pickle.
2.  **Enrichment**: `processing.enrich_sessions(sessions)` adds derived cols (e.g., `velocity_px_s` if missing).
3.  **Filtering**: `processing.filter_post_injection(sessions)` (AMPH only) cuts data to the post-drug window.
4.  **Signal Extraction**: Code looks up the brain region in `meta` to find the correct `signal_X_zscore` column.
5.  **Event Triggering**: Code uses `df['onset']` or `df['syllable']` transitions to slice `df['signal_X']` windows.

## 6. Pickle Variations (Dirty vs Sanitized)
The session pickles come in different "flavors" depending on how they were built (e.g., via `create_jonathan_sessions.py`).

### Dirty Pickle (`_dirty.pkl`)
*   **Created with**: `FORCE_DIRTY=True`.
*   **Content**: Contains **ALL** data, ignoring manual exclusions or auto-masking.
*   **Use Case**: Debugging raw data, checking why data might be excluded, or verifying coverage of exclusions.
*   **Warning**: Do not use for final biological analysis as it includes bad data (artifacts, patch cord failures).

### Sanitized Pickle (`_sanitized.pkl` or default)
*   **Created with**: `FORCE_DIRTY=False` (Default).
*   **Content**: Excludes bad data. Frames marked for exclusion (e.g., high motion artifact, lost tracking) are set to `NaN` or removed.
*   **Use Case**: Standard for all scientific analysis.

### "w_sylls" (`_w_sylls.pkl`)
*   **Definition**: Explicitly indicates that **Keypoint-MoSeq syllables** have been attached to the dataframe.
*   **Pipeline**:
    1.  Base sessions are created (kinematics + photometry).
    2.  `attach_syllables_kpms.py` loads the base pickle + `results.h5` (MoSeq output).
    3.  It merges the syllable IDs into `s.all_data_df` and saves the new file with `_w_sylls` suffix.
*   **Note**: All analysis modules (PPE, Entropy, Panel 1z) require `w_sylls` files.
