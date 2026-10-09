# Syllable Weaving Strategy

## 1. Overview
This document details the architectural shift from a **Filtered** output (where `NaN`s are deleted) to a **Full-Session / Weaved** output (where `NaN`s are preserved in a full-length dataframe).

*   **Old Strategy**: Filter Photometry to match valid Tracking frames.
    *   *Result*: Clean dataset, but lost time information (gaps removed).
*   **New Strategy**: **Syllable Weaving**.
    *   *Result*: Maintain the original, absolute Photometry timeline. "Weave" valid syllables into it. Gaps in tracking become `NaN`s in the Syllable columns.

## 2. Why "Weave" (Compute First, Join Later)?
Keypoint-MoSeq (the modeling library) requires contiguous, valid data to function correctly (PCA, AR-HMM). It cannot handle `NaN`s natively.
Therefore, we must:
1.  **Extract Valid Chunks**: Use VFI to pull only valid data.
2.  **Compute Syllables**: Run the model on these valid, condensed chunks.
3.  **Weave Back**: Take the resulting syllables and place them back onto the original timeline using the VFI as a "coordinate map".

## 3. Sampling Rates & Upsampling
The weaving logic differentiates between projects based on the native recording rate:

*   **IFDN (80Hz Native)**:
    *   **VFI**: Absolute 80Hz indices (Step = 1).
    *   **Model**: 80Hz output.
    *   **Logic**: 1:1 Mapping. No extra `ffill` required for upsampling.
*   **AMPH (40Hz Native -> 80Hz Weaved)**:
    *   **VFI**: Originally 40Hz (detected by `migrate_legacy_vfi`).
    *   **Scaling**: Indices are doubled (Step = 2) to match 80Hz timeline.
    *   **Logic**: 1:2 Expansion. Model frame `i` covers two 80Hz frames via `ffill`.

## 4. Interpolation & NaN Expectations
We use **Linear Interpolation (`limit=4`)** on the Photometry.
*   **NaN Signature**: Peaks at NaN run lengths **7** and **3** occur in both projects. This is a mathematical artifact of the `limit=4` interpolator bridging small gaps caused by 40Hz-locked auto-masking on an 80Hz timeline.
*   **Verdict**: These specific NaN patterns are **EXPECTED** and verified as safe.

## 5. Summary Workflow
1.  **Mother Index**: Initialize dataframe to length `n_frames` (Video Reference).
2.  **Scale VFI**: Convert relative/legacy VFI to absolute 80Hz indices.
3.  **Place**: Map model results onto the timeline using scaled VFI coordinates.
4.  **FFill**: Apply forward fill to syllables (to cover 1-frame gaps in 40Hz projects).
5.  **Output**: `all_data_df` containing aligned behavioral and neural data.
remember maybe we can just generate good vfis from the start using trimming.py
we dont use vfi to pull valid data we only use it for weaving things back
finding some good tests is good to consider

also maybe we need more folder specific diagnostics (like we do config and utils atm) in general.