# Behavioral Strategy Modeling Pipeline - Status Update (Monday)

This document provides a summary of the current status of the Behavioral Strategy Modeling pipeline as of Friday, June 19, 2026, to serve as a checkpoint for resuming work.

---

## 1. Where We Are At (Current Progress)

We have successfully built and verified the core infrastructure for **both the micro-level fusion and macro-level strategy layers** of the pipeline, using the GPU-accelerated cuML framework on all 15 `ifdn` sessions.

### Micro-Level: Kinematic Sub-State Clustering
*   **Bout Extractor:** Slices raw frame sequences into continuous syllable blocks (excluding `-1` background frames) and extracts 5 features: `duration_frames`, `mean_velocity`, `mean_angular_velocity`, `mean_acceleration`, and `mean_dist_from_center`.
*   **Clustering Engine:** Fits local K-Means models per syllable using GPU-accelerated **RAPIDS cuML** on the pooled bouts across all sessions.
*   **Vocabulary Expansion:** Maps the 98 base syllables into **260 unique Rich Tokens** (sub-states representing posture + kinematic execution regime).

### Macro-Level: Sliding-Window Strategy Topic Modeling
*   **Sliding Windows:** Chunks the frame-aligned Rich Token sequence into sliding windows of $128\text{s}$ (with a stride of $10\text{s}$) to extract slow-varying dynamics. Window contents are treated as document "bags-of-words".
*   **Topic Model (LDA):** Fits an unsupervised Latent Dirichlet Allocation (LDA) model to cluster the $4,315$ windows into 5 distinct behavioral **Strategy IDs**.
*   **Frame-level Mapper:** Vectorized closest-center projection maps the strategy IDs back to 80Hz frame-level sequences aligned with the original dataframe length (0 unassigned frames).

---

## 2. Key Files Implemented

1.  **Micro-Fusion Module:** [clustering.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/fusion/clustering.py) — Contains bout extraction and `KinematicSyllableClustered` model.
2.  **Macro-Strategy Model:** [strategies.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/models/strategies.py) — Contains `StrategyTopicModel` (LDA) and diagnostic composition helpers.
3.  **Preprocessing Helpers:** [segmentation.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/preprocessing/segmentation.py) — Modified to add `generate_sliding_windows` and `assign_frame_strategies`.
4.  **Automatic Notification Wrapper:** [run_with_ntfy.sh](file:///home/fsp585/.gemini/antigravity-ide/brain/9532cdcd-4e72-4b26-9fee-372b0a13f10a/scratch/run_with_ntfy.sh) — Script to run the pipeline in `nohup` mode and send start/fail/success push notifications to your `ntfy` channel (`https://ntfy.sh/woody_videos_seb`).

---

## 3. Discovered Visualizations

We ran the pipeline and output diagnostics to `/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/`:

*   **Syllable Kinematics PCA Plots:** Projected scaled kinematic features for the top 3 most frequent syllables (Syllables 53, 4, and 41) into 2D to show sub-state clusters.
*   **Window Strategy UMAP:** Reduced the window term-frequency representations into 2D. Displays clear, separated window groupings corresponding to Strategy IDs.
*   **Session Barcode Timelines:** Plotted Strategy IDs frame-by-frame over time. This visually confirms that **strategies change slowly and smoothly** over minutes (macroscopic states) rather than rapidly switching frame-to-frame.

*You can view all the generated plots embedded in:*
*   [walkthrough.md](file:///home/fsp585/.gemini/antigravity-ide/brain/9532cdcd-4e72-4b26-9fee-372b0a13f10a/walkthrough.md)

---

## 4. Next Steps (Monday's Agenda)

We are ready to move on to **Phase 3 (The Biological "Go/No-Go" Gate)** to validate that the strategy clusters represent true neurobiology:

1.  **Dopamine (dLight) PETH Alignment:**
    *   Locate the timestamps/frames where Strategy transitions occur.
    *   Align striatal dLight signals (e.g. `signal_0_zscore_datta`) around a $[-15\text{s}, +15\text{s}]$ window at transition boundaries.
    *   Verify whether strategy shifts correspond to significant peaks or dips in average dopamine levels.
2.  **Pharmacological Entropy Collapse:**
    *   Compute the Shannon entropy of strategy ID distributions.
    *   Test if Amphetamine-treated groups show a significant reduction in entropy compared to Saline controls, indicating a collapse into stereotyped strategies.
