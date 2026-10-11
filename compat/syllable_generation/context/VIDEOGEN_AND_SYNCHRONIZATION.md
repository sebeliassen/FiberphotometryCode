# VideoGen & Synchronization Verification

**Goal**: Explain the architecture of the Video Generation module (`videogen`) and how it serves as the primary verification tool for Syllable-Video synchronization.

## 1. Overview
The `videogen` module produces composite videos (e.g., Side-by-Side or Trio layouts) that visualize the alignment between:
1.  **Behavior Video** (Raw AVI)
2.  **Syllable Data** (Keypoint-MoSeq output)
3.  **Photometry Signals** (dF/F traces)

It is the visual proof that the "Syllable Weaving" strategy works as intended.

## 2. Synchronization Architecture
The underlying data comes from the "Weaved" pickles (`w_sylls.pkl`).
*   **The Invariant**: Efficiency is secondary to alignment. The dataframe provided to `videogen` is **Time-Locked** to the video frames.
*   **1:1 Mapping**: `df.iloc[i]` corresponds exactly to Video Frame `i`. This is achieved by anchoring the dataframe index to the video's `n_frames` metadata.
*   **Missing Data**: Frames where tracking or photometry is missing are preserved as `NaN` rows, keeping the absolute timeline intact.

## 3. Visual Verification Tools
To empirically prove this synchronization, the module offers specific verification flags:

### Tracking Overlays (`--show-tracking`)
Draws raw tracking data from the dataframe directly onto the video grid.
*   **Red Dot**: `centroid_x`, `centroid_y` from the dataframe.
*   **Green Arrow**: `heading` from the dataframe.
*   **Verification**: If the Red Dot is on the mouse, the **Syllable is Valid**. (Since Syllable and Centroid share the same row/index).

### Heading Correction (`--flip-heading`)
*   **Problem**: Coordinate system mismatches (e.g., Model 0° = Right, CV2 0° = Left) can cause the heading arrow to point backwards.
*   **Solution**: The `--flip-heading` flag adds 180° ($\pi$ radians) to the visualization angle. It is a display-side correction that does not alter data.

## 4. Key Logic Concepts

### A. Sticky Syllable Display
To prevent visual flickering when a specific syllable's trajectory plot (PDF) is missing:
*   **Logic**: If `Syllable X` is requested but `SyllableX.pdf` is missing, the generator continues to display the **Last Valid Syllable Image**.
*   **Benefit**: Maintains visual continuity during rapid transitions or sparse plot coverage.

### B. Real-Speed Playback
Because the source data (80Hz) often exceeds standard video framerates (30Hz), a 1:1 render results in slow-motion output.
*   **Sampling**: The composer calculates a `stride = Source_FPS / Target_FPS` (e.g., 2.67).
*   **Execution**: It samples every ~$N$-th frame from the dataframe and seeks the video player to match.
*   **Result**: 1 second of experiment time equals 1 second of video playback.

## 5. Usage Reference

**Standard Verification Command:**
```bash
python syllable_generation/videogen/main.py \
    --pkl <path_to_pkl> \
    --output <dir> \
    --session 0 \
    --trio \
    --show-tracking \
    --flip-heading \
    --test-syllable-mode
```

*   `--trio`: Generates the full 3-panel layout.
*   `--test-syllable-mode`: Automatically finds the most interesting window (many syllable changes) to render.
