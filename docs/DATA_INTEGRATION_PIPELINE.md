# System Architecture: Multi-Modal Data Synchronization using the "High-Res Anchor"

## 1. Executive Summary: The "High-Res Anchor"
The central philosophy of this codebase is that **Video Time is Absolute**.
Regardless of how data is sampled (Photometry @ ~80Hz, Models @ 40Hz), everything must eventually map back to the unique integer indices of the original video frames ($t=0, 1, 2...$).

We **never downsample the ground truth** (Photometry/Video) to match a model. We **upsample/tile the model** to match the ground truth.

---

## 2. Data Topology: Where does data live?

A common confusion is "Where is the database?". This project uses a **Filesystem-based Object Database**.

### A. The Raw Input (`src_oft/`)
The entry point is a directory (e.g., `src_oft/jonathan_amph_workspace`) containing unstructured files:
*   `*.mp4` (Video)
*   `*.csv` (Photometry traces)
*   `*.slp` / `*.h5` (Tracking output from SLEAP/DLC)

### B. The Object Store (`loaders/*.pkl`)
We do not query a SQL database. Instead, we serialize massive lists of Python `Session` objects into Pickle files.
1.  **`_dirty.pkl`**: Crude scan of the raw folder. May contain broken/incomplete sessions.
2.  **`_sanitized.pkl`**: Cleaned sessions. Photometry is attached, timestamps are aligned to the Anchor, manual exclusions are applied. **This is the clean 80Hz dataset.**
3.  **`_w_sylls_final.pkl`**: The Final Master. Contains everything in `_sanitized` + merged 40Hz behavioral syllables.

---

## 3. The Synchronization Logic (Deep Dive)

### The Problem: "Time Warp"
*   **Video:** 80 FPS. (Frame 0, 1, 2, 3...)
*   **Model:** 40 FPS. (Step 0, 1...)
*   **Risk:** If you map Model Step 1 to Video Frame 1, you effectively play the behavior at **2x speed**, desynchronizing it from the neural signal.

### The Solution: Downsample Rate Detection & Tiling
We use a **Metadata-Driven** approach to resolve this.

1.  **Detection:**
    *   We read `model_summary.json` (from the model training folder) to find `downsample_rate` (e.g., `2.0`).
    *   *Why not guess?* Guessing based on length is dangerous if sessions vary in duration.

2.  **Index Scaling:**
    *   Model Index $i$ corresponds to Anchor Frame $\text{round}(i \times \text{Rate})$.
    *   *Example:* Model Step 100 $\rightarrow$ Anchor Frame 200.

3.  **Tiling (The Zero-Order Hold):**
    *   Since Behavior is categorical (Syllable A vs B), we cannot smooth it.
    *   We **Forward Fill** (`ffill`) the gaps created by upsampling.
    *   *Example (Rate=2):*
        *   Frame 200: Syllable A (From Model)
        *   Frame 201: Syllable A (Filled)
        *   Frame 202: Syllable B (From Model)

---

## 4. Pipeline Modules & Key Functions

This section lists the essential modules that drive the pipeline, in order of execution.

| Module / Script | Key Functionality |
| :--- | :--- |
| **`loaders/jonathan/create_jonathan_amph.py`** | **The Entry Point.** Orchestrates the initial build. Calls `build.py` to scan `src_oft` and saves `_dirty.pkl`. |
| **`loaders/jonathan/steps/fiberphotometry.py`** | **Signal Parser.** Specifically `realize_photometry_columns` aligns raw CSV timestamps to Video Frame Indices (Ground Truth). |
| **`projects/jonathan_openfield_amph/enrich.py`** | **Manual Exclusions.** `apply_manual_exclusions` applies the mask from `exclusions.csv` to the 80Hz photometry data. |
| **`analysis/generate_metadata_csv.py`** | **Metadata Gen.** Scans session IDs to generate `amph_session_metadata.csv` (Drug, Dose, Genotype). |
| **`analysis/create_amph_master_df.py`** | **The Master Builder.** <br>1. Loads `_sanitized.pkl` (80Hz).<br>2. Loads Model Results (40Hz).<br>3. **Detects Rate** & Scales Indices (x2).<br>4. **Tiles** Syllables (ffill).<br>5. Merges everything into `_w_sylls_final.pkl`. |
| **`analysis/replications/datta_utils.py`** | **The Merger.** `build_aligned_moseq_phot` handles the specific Pandas logic of Left-Joining Photometry onto Syllables. |
| **`projects/.../cleaning_gui.ipynb`** | **Visualization.** `DraggableRangeGUI`. Allows human verification. Crucially, it scales Auto-Exclusions by reading `model_summary.json` so overlays match the 80Hz plot. |

---

## 5. Critical Artifact Reference

To debug alignment, you must understand these three files:

| Artifact | Location | Role |
| :--- | :--- | :--- |
| **`results.h5`** | Model Folder | Contains the raw **Latent States** and **Syllables**. It knows "WHAT" happened, but has lost the absolute "WHEN". |
| **`video_frame_indexes.npz`** | Model Folder | The Rosetta Stone. Contains the mappings that link `results.h5` rows back to the original video frames. **This is what we scale.** |
| **`exclusions.csv`** | Project Overrides | A manual list of "Bad Data" intervals. **Crucially:** These indices refer to the **80Hz Anchor**, not the model time. |

---

## 6. FAQ: Edge Cases & "What Ifs"

### Q1: What happens if the frame counts don't align perfectly?
**A:** The system permits small "jitter". The `datta_utils.combined_photometry_and_syllables` function uses linear interpolation to bridge gaps of a few frames (e.g., dropped samples). However, if there is a gross mismatch/key error, the loader will raise an exception during the `build` phase.

### Q2: I added a new mouse. Why isn't it showing up?
**A:** The system does not "watch" folders. You must re-run the `build.py` or `create_jonathan_amph.py` script. This triggers `tracking.discover_sessions`, which scans the `src_oft` directory and updates the `_dirty.pkl`.

### Q3: Why do we modify `create_master_df` for every cohort?
**A:** Because directory structures and naming conventions (`src_oft` vs `src_new`) change. The **Logic** (scaling/tiling) is universal, but the **Paths** are specific. You should copy the script and update the `PROJECT_DIR` and `SESSIONS_PKL_IN` constants.

### Q4: How do I verify the alignment visually?
**A:** Use the `cleaning_gui.ipynb`. We updated it to read the `downsample_rate` metadata. If, for example, the "Green" valid signal overlay stops halfway through the session, your downsample rate is wrong (the "Time Warp" bug). If it covers the full session, alignment is correct.
