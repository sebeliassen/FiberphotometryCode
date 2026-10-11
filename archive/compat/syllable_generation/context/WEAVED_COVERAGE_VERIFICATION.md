# Weaved Coverage Verification

**Goal**: Establish the verification status of "Weaved" 1-to-1 Photometry pickles and define the correct diagnostic logic for future data.

## 1. Overview
As of Jan 2026, the pipeline uses the **Syllable Weaving** strategy, where full-session photometry pickles are saved (preservings gaps/NaNs). Verification confirms that manual exclusions from `exclusions.csv` are applied correctly to these absolute timelines.

## 2. The "Direct Indexing" Invariant
A key finding during verification (covering both IFDN and AMPH projects) is that the manual exclusion indices refer to **Absolute Photometry Frames** directly.

*   **Logic**: Exclusions (`idx:START-END`) map 1:1 to the dataframe indices.
*   **No Translation**: Unlike older "Filtered" streams, these indices do **NOT** require translation via VFI (Video Frame Index).
*   **Scaling**: Since the pickles are 80Hz "Weaved", no further scaling (e.g., 40Hz -> 80Hz) should be applied to the CSV indices.

## 3. Verification Status (Baseline Jan 9, 2026)
Formal verification was conducted on the latest production pickles using `verify_weaved_coverage.py`.

| Project | Target Pickle | Pass Rate (Baseline) | Mode | 
| :--- | :--- | :--- | :--- |
| **IFDN** | `jonathan_ifdn_sessions_w_sylls.pkl` | **100% (12/12)** | `--direct --disable-scale` |
| **AMPH** | `jonathan_amph_sessions_w_sylls.pkl` | **100% (21/21)** | `--direct --disable-scale` |

### **Cardinality & Anchoring (The "Mother Index")**
A significant resolution was achieved regarding session lengths. 
*   **The Problem**: Some IFDN sessions were shorter than their video recording because they were clipped to the tracking end (VFI Max).
*   **The Fix**: All standard pickles are now **Anchored to `n_frames`** from the primary video reference.
*   **Result**: The dataframe index is identical to the video frame index. Gaps at the start or end are preserved as `NaN` syllables, ensuring absolute time-locking.

### Sensitivity Results
Both projects passed sensitivity checks (checking data immediately after exclusions). Failures in sensitivity metrics were attributed to session boundaries (end-of-file padding) or natural tracking sparsity, not alignment errors.

## 4. Current Tooling & SOP
The following tools are the only verified diagnostics for weaved pickles. All others in the `diagnostics/` folder have been deprecated.

### Standard Verification Command
To verify data integrity for new pickles:
```bash
python3 syllable_generation/diagnostics/verify_weaved_coverage.py \
    --pkl <path_to_pkl> \
    --csv <path_to_exclusions_csv> \
    --direct \
    --disable-scale \
    --verbose
```

### Critical Files
*   `diagnostics/verify_weaved_coverage.py`: Core verification tool.
*   `diagnostics/run_weaved_coverage_matrix.sh`: Matrix testing wrapper (Baseline + Sensitivity).

## 5. Summary for Future Work
*   Always use `--direct` when verifying exclusions for IFDN/AMPH.
*   If a 0% pass rate is encountered, check if the project has reverted to a filtered-VFI paradigm (unlikely for Jan 2026+ data).
*   Trust the 1:1 cardinality of Weaved pickles; gaps are NaNs, not missing rows.
