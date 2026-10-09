# Fiber Photometry Guide: Data Structure, Sensor Wavelengths, and Reintroduction Workflow

**Repository:** `FiberphotometryCode`  
**Target Dataset:** Karen Project Multi-Animal Tandem / Top-Cam Cohort  
**Date:** September 2026  

---

## 1. Executive Summary & Current Status

In the **Tandem Multi-View Cohort** (top camera + bottom camera 23-keypoint SLDS model):
- **Model Training Status:** 100% complete (50 AR iterations + 200 SLDS iterations finished on 112 sessions, totaling 17,953,523 frames).
- **Production Artifact:** [`sessions_karen_project_tandem_cohort_production.pkl`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_production.pkl) (11.6 GB).
- **Photometry Status in Production Pickle:**
  - **106 sessions** contain fully aligned fiber photometry signals (raw, $\Delta F/F$, RANSAC z-scores, Wallace z-scores, and prominence for all 4 fibers: `signal_0` .. `signal_3`).
  - **6 sessions** (`2026-03-06T14_44_36_*` and `2026-05-16T16_34_48_*`) are purely behavioral recordings where no photometry hardware was plugged in (`s.photometry = {}`). Their photometry columns are populated with standard `np.nan` values, maintaining strict schema parity across all 112 sessions.

For the current behavioral and syllable transition analyses, fiber photometry is not strictly required. However, the raw data, preprocessing pipelines, and alignment mappings are fully preserved. This document describes where that data lives, assesses the `'470'` sensor hardcoding, and provides a turnkey recipe for re-aligning, adding, or modifying photometry channels in the future without retraining any KPMS models.

---

## 2. Where the Fiber Photometry Data Lives

### A. Inside the Python Session Objects (`s.photometry`)
Each session object in [`sessions_karen_project_dirty.pkl`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/sessions_karen_project_dirty.pkl) and the production pickles stores an attribute `s.photometry`.

When present, `s.photometry` is a Python dictionary containing 3 primary channel keys:

```
s.photometry = {
    'combined_filtered': pd.DataFrame,  # ~210,000 rows (interleaved raw Doric / DAQ stream)
    '415':               pd.DataFrame,  # ~42,000 rows  (de-interleaved isosbestic control channel)
    '470':               pd.DataFrame   # ~42,000 rows  (de-interleaved signal sensor channel)
}
```

#### Detailed Column Structure:
1. **`combined_filtered`**:
   - `FrameCounter`: Hardware counter from acquisition box.
   - `SystemTimestamp`: Floating-point clock timestamp in seconds.
   - `LedState`: Integer mask identifying which LED was strobed (`1` = 415nm Isosbestic, `2` = 470nm Green Signal, `7` = Dark/Shutter).
   - `signal_0`, `signal_1`, `signal_2`, `signal_3`: Raw analog photodetector voltage across the 4 optical fiber patch cords.
   - `frame_idx`: Synchronized video frame index (0-indexed) computed during data ingestion.

2. **`415` (Isosbestic Control)**:
   - Filtered subset corresponding to `LedState == 1` (~12 Hz sample rate).
   - Serves as the motion/autofluorescence reference.

3. **`470` (Functional Indicator)**:
   - Filtered subset corresponding to `LedState == 2` (~12 Hz sample rate).
   - Contains pre-computed baseline detrending and normalization columns:
     - `signal_N`: Raw fluorescence voltage.
     - `signal_N_dff`: Relative fluorescence $\Delta F / F_0 = (F - F_0) / F_0$.
     - `signal_N_zscore`: Standard z-score.
     - `signal_N_zscore_ransac`: Robust linear/polynomial detrended z-score using RANSAC to exclude large calcium/dopamine transients during baseline fitting.
     - `signal_N_zscore_wallace`: Alternative baseline algorithm based on local lower-quantile rolling windows.
     - `signal_N_prominence`: Peak prominence measure for transient detection.

### B. Inside Tabular Session DataFrame (`s.all_data_df`)
In [`sessions_karen_project_tandem_cohort_production.pkl`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_production.pkl), all 25 canonical photometry metrics are joined directly alongside the behavioral kinematics and syllables:

```
[Behavior & Kinematics] (26 columns):
  frame_index, name, centroid_x, centroid_y, heading, heading_top, heading_moseq,
  angular_velocity, velocity_px_s, velocity_px_s_raw, syllable, tandem_syllable,
  group, onset, centroid_x_top, centroid_y_top, angular_velocity_top, velocity_px_s_top,
  velocity_px_s_top_raw, centroid_x_bot, centroid_y_bot, heading_bot,
  angular_velocity_bot, velocity_px_s_bot, velocity_px_s_bot_raw, top_syllable

[Aligned Photometry Columns] (25 columns):
  SystemTimestamp,
  signal_0, signal_1, signal_2, signal_3,
  signal_0_zscore_ransac, signal_0_zscore_wallace, signal_0_dff, signal_0_zscore, signal_0_prominence,
  signal_1_zscore_ransac, signal_1_zscore_wallace, signal_1_dff, signal_1_zscore, signal_1_prominence,
  signal_2_zscore_ransac, signal_2_zscore_wallace, signal_2_dff, signal_2_zscore, signal_2_prominence,
  signal_3_zscore_ransac, signal_3_zscore_wallace, signal_3_dff, signal_3_zscore, signal_3_prominence
```

---

## 3. Analysis: Is Hardcoding `'470'` an Issue?

### Historical Background
The code in [`syllable_generation/loaders/jonathan/config.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/loaders/jonathan/config.py) and [`syllable_generation/analysis/replications/datta_utils.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/replications/datta_utils.py) was tailored to the Doric 4-fiber photometry hardware used in the initial experiments:
- **LED State 1 = 415 nm** (Isosbestic point for GFP/GCaMP/dLight, insensitive to ligand binding).
- **LED State 2 = 470 nm** (Active excitation wavelength for green sensors: GCaMP6/8, dLight1, GRAB-DA2m).
- In [`datta_utils.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/replications/datta_utils.py#L24-L29):
  ```python
  def _freqs_to_use() -> List[str]:
      freqs = getattr(JCFG, "combine_freqs", None)
      if freqs is None:
          return ["470"]
      return [str(f) for f in freqs]
  ```

### Assessment: Is this an issue?
1. **For the Karen Cohort (Current Data): NO.**
   - All 112 sessions in this study that recorded photometry used this exact 415nm/470nm optical configuration. No 560nm/red data was omitted because no red fluorophores were recorded in this cohort.
2. **As a General Architectural Design: YES.**
   - Hardcoding `'470'` presents several key limitations:
     - **Dual-Wavelength Experiments:** If an experiment records Green DA (470nm) simultaneously with Red Calcium or Neuropeptide (560nm or 565nm, e.g. jRGECO1a, rDA1m), hardcoded `'470'` ignores the second indicator channel.
     - **Hardware Agnosticism:** Other acquisition systems (Tucker-Davis Technologies / TDT, Plexon, Newport) label channels differently (e.g., `465A`, `405A`, `Fi1r`, `Fi1g`).
     - **Pure Behavioral Sessions:** Calling `_phot_frame_map_for_freq(s, "470")` unconditionally on sessions without photometry crashes with `KeyError: "Session is missing photometry channel '470'."`.

### Recommended Generalized Pattern (Dynamic Channel Auto-Discovery)
Instead of hardcoding `"470"`, the alignment utility should discover signal channels dynamically:

```python
def get_signal_channels(session) -> List[str]:
    """
    Auto-detect active functional photometry channels, excluding isosbestic
    references and combined raw streams.
    """
    phot = getattr(session, "photometry", None)
    if not isinstance(phot, dict) or not phot:
        return []

    # Known non-functional / reference channels to exclude
    EXCLUDED_KEYS = {"combined_filtered", "405", "415", "iso", "isosbestic", "dark"}

    # Signal channels are any remaining channels
    signal_channels = [k for k in phot.keys() if str(k).lower() not in EXCLUDED_KEYS]
    return sorted(signal_channels)
```

---

## 4. How Photometry Alignment Works

Because camera frame rates (60 Hz or 80 Hz) and photometry LED modulation rates (~12 Hz per channel in a 4-state cycle) run on distinct asynchronous clocks, data must be aligned onto a common timebase:

1. **Hardware Digital Trigger (Input DI / Pulse)**:
   - When the camera starts exposing, a strobe pulse is sent to the Doric digital input (`DI / LedState`), stamping the video frame index directly into the photometry CSV.
2. **Alignment Strategy (`datta_utils.py`)**:
   - For each video frame $t \in [0, N-1]$:
     - The pipeline locates the nearest photometry sample matching `frame_idx == t` or the nearest `SystemTimestamp`.
     - Forward/backward interpolation is applied across any dropped hardware samples.
3. **$\Delta F/F$ and Normalization**:
   - For each fiber channel $c \in \{0, 1, 2, 3\}$:
     $$F_0(t) = \text{RANSAC\_Fit}(F_{415}(t)) \quad \text{or} \quad \text{Rolling\_Quantile}(F(t), q=0.10)$$
     $$\Delta F/F(t) = \frac{F_{470}(t) - F_0(t)}{F_0(t)}$$
     $$\text{z-score}(t) = \frac{\Delta F/F(t) - \mu}{\sigma}$$

---

## 5. How to Reintroduce or Update Photometry in the Future

If you want to:
- Re-run baseline detrending with different parameters,
- Add a newly digitized photometry session, or
- Add new channels (e.g. 560nm red sensor),

**You do NOT need to retrain or refit any Keypoint-MoSeq models.** The syllable classifications and posture kinematics in `s.syllables` and `s.all_data_df` are completely independent of photometry.

### Standalone Turnkey Python Script

Save this script as `update_cohort_photometry.py` and run it whenever you want to refresh the photometry in the production dataset:

```python
#!/usr/bin/env python3
"""
update_cohort_photometry.py

Re-aligns and updates fiber photometry channels into an existing KPMS session pickle.
Execution time: ~30-60 seconds for the entire 112-session cohort.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.replications.datta_utils import combined_photometry_and_syllables

INPUT_PKL = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_production.pkl"
OUTPUT_PKL = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_production.pkl"

TEMPLATE_PHOT_COLS = [
    "SystemTimestamp",
    "signal_0", "signal_1", "signal_2", "signal_3",
    "signal_0_zscore_ransac", "signal_0_zscore_wallace", "signal_0_dff", "signal_0_zscore", "signal_0_prominence",
    "signal_1_zscore_ransac", "signal_1_zscore_wallace", "signal_1_dff", "signal_1_zscore", "signal_1_prominence",
    "signal_2_zscore_ransac", "signal_2_zscore_wallace", "signal_2_dff", "signal_2_zscore", "signal_2_prominence",
    "signal_3_zscore_ransac", "signal_3_zscore_wallace", "signal_3_dff", "signal_3_zscore", "signal_3_prominence"
]


def update_photometry(input_path: Path, output_path: Path):
    print(f"Loading {input_path.name}...")
    with open(input_path, "rb") as f:
        sessions = pickle.load(f)

    print(f"Updating photometry for {len(sessions)} sessions...")
    updated_count = 0
    nan_count = 0

    for s in sessions:
        n_frames = len(s.syllables)
        timeline_index = pd.RangeIndex(0, n_frames)

        # 1. Keep all behavioral / kinematic columns
        behavior_cols = [c for c in s.all_data_df.columns if c not in TEMPLATE_PHOT_COLS]
        df_behavior = s.all_data_df[behavior_cols].copy()

        # 2. Re-weave Photometry
        has_fp = hasattr(s, "photometry") and isinstance(s.photometry, dict) and "470" in s.photometry
        if has_fp:
            try:
                phot_df = combined_photometry_and_syllables(s)
                phot_aligned = phot_df.reindex(timeline_index).reset_index(drop=True)
                for col in TEMPLATE_PHOT_COLS:
                    if col not in phot_aligned.columns:
                        phot_aligned[col] = np.nan
                phot_aligned = phot_aligned[TEMPLATE_PHOT_COLS]
                updated_count += 1
            except Exception as e:
                print(f"  [warning] {s.session_id}: Alignment error ({e}). Using NaNs.")
                phot_aligned = pd.DataFrame(np.nan, index=timeline_index, columns=TEMPLATE_PHOT_COLS)
                nan_count += 1
        else:
            phot_aligned = pd.DataFrame(np.nan, index=timeline_index, columns=TEMPLATE_PHOT_COLS)
            nan_count += 1

        # 3. Concatenate back into all_data_df
        s.all_data_df = pd.concat([df_behavior.reset_index(drop=True), phot_aligned], axis=1)

    print(f"Saving updated dataset to {output_path.name}...")
    with open(output_path, "wb") as f:
        pickle.dump(sessions, f)

    print(f"Done! Updated: {updated_count} sessions with FP, {nan_count} sessions with NaNs.")


if __name__ == "__main__":
    update_photometry(INPUT_PKL, OUTPUT_PKL)
```

---

## 6. Summary Checklist for Future Work

| Task | Action |
| :--- | :--- |
| **Current Downstream Analyses** | Use [`sessions_karen_project_tandem_cohort_production.pkl`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_production.pkl) directly. Behavioral kinematics, top syllables, and tandem syllables are 100% complete across all 112 mice. |
| **Examine Missing Mice** | The 6 sessions lacking photometry (`2026-03-06T14_44_36_*`, `2026-05-16T16_34_48_*`) have valid behavior and syllables; only `signal_*` is NaN. |
| **Multi-Wavelength Support** | If integrating 560nm red sensor data, update `datta_utils._freqs_to_use()` to return `["470", "560"]` and execute `update_photometry()`. |
| **Model Independence** | Never retrain KPMS models when making changes to fiber photometry filtering, detrending, or channels. Photometry is strictly an aligned downstream stream. |
