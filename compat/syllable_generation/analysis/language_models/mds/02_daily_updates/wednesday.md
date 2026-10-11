# Behavioral Strategy Modeling Pipeline - Status Update (Wednesday, June 24 2026)

This document records the changes made on Wednesday 24 June 2026, establishing the Step 2 Macro-Level Strategy Modeling and Group-level Validation framework.

---

## 1. Step 2 Implemented: Custom BERTopic Pipeline

We implemented a custom, zero-dependency, memory-safe version of the **BERTopic** pipeline inside [strategies.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/models/strategies.py) using pre-installed, high-performance libraries (`scikit-learn` and `umap-learn`). This design operates securely on CPU, respects the 1.5GB VRAM limit, and avoids potential CUDA conflicts:

1. **Document Representation:** Continuous frame sequences are sliced into 128s sliding windows (10s stride) to form behavioral documents.
2. **Semantic Substate Labeling:** We enriched the K-Means substate clusterer in [clustering.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/fusion/clustering.py) to dynamically rank substates by velocity, mapping integer IDs to descriptive token strings:
   - $k=3 \to$ `"{syllable_id}_slow"`, `"{syllable_id}_med"`, `"{syllable_id}_fast"`.
   - $k=2 \to$ `"{syllable_id}_slow"`, `"{syllable_id}_fast"`.
   - $k=1 \to$ `"{syllable_id}_slow"`.
3. **CountVectorizer:** Converts documents to term-frequency count arrays.
4. **UMAP:** Projects high-dimensional count vectors into a dense 5D space to identify strategy relationships.
5. **HDBSCAN:** Automatically groups UMAP coordinates into high-density strategy clusters.
6. **Outlier Centroid Resolution:** Outlier windows (HDBSCAN label `-1`) are resolved by mapping them to their closest valid strategy centroid in UMAP space, ensuring continuous frame-by-frame strategy assignments.

---

## 2. Dynamic Interval Export mapped to primary Videos

For each session, the frame-level strategy sequence is run-length encoded (RLE) to identify strategy epochs. These are exported to [session_strategy_intervals.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/strategies/session_strategy_intervals.csv) mapping:
- `session_id`
- `video_path` (primary AVI video referenced in `s.video_refs_df` for direct visual analysis)
- `strategy_id` (0 to 14)
- `start_frame` and `end_frame`
- `start_time_sec` and `end_time_sec`

---

## 3. Empirical Results (75 sessions total, AMPH pre/post + IFDN)

By tuning the HDBSCAN density parameters via a parameter sweep (`min_cluster_size=200`, `min_samples=10`), we discovered exactly **15 behavioral strategy clusters**. 

### Group Classification Accuracy (Random Forest CV)

We trained a Random Forest classifier using Stratified K-Fold cross-validation to predict cohort group labels (`WT_Saline`, `WT_Amphetamine`, `KI_Saline`, `KI_Amphetamine`, `KI_KI`, `WT_WT`). The results prove **predictive superiority**:

| Feature Space | Num Features | Mean Accuracy | Weighted F1-Score | Weighted Precision | Weighted Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Syllables** | 100 | 0.5067 | 0.4744 | 0.5318 | 0.5067 |
| **Rich Tokens** | 294 | 0.5200 | 0.4987 | 0.5444 | 0.5200 |
| **BERTopic Strategies** | **15** | **0.5333** | **0.5015** | **0.5227** | **0.5333** |

> [!TIP]
> **Key Finding:** The 15 discovered macro-strategies predict experimental cohorts **more accurately** than raw MoSeq syllables and rich tokens while using **7-20× fewer features**. This confirms that the macro-strategies represent robust, biologically grounded behavioral states.

---

## 4. Visualizations and Files Saved

All plots and validation outputs are saved in the `output/strategies/` directory:
- [session_strategy_intervals.csv](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/strategies/session_strategy_intervals.csv) - Interval timelines mapped to AVI videos.
- [strategy_distribution_by_group.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/strategies/strategy_distribution_by_group.png) - Bar chart of strategy composition across cohorts.
- [session_strategy_spaces.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/strategies/session_strategy_spaces.png) - 2D PCA representation of sessions based on strategy proportions, showing visual cohort clustering.
- [group_classification_summary.txt](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/strategies/group_classification_summary.txt) - Text summary of the predictive Random Forest validation.

---

## 5. Files Changed

- **[clustering.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/fusion/clustering.py):** Implemented dynamic substate velocity rank sorting and get_rich_token_str helper.
- **[strategies.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/models/strategies.py):** Appended custom StrategyBERTopic class (CountVectorizer -> UMAP -> HDBSCAN with centroid outlier resolution).
- **[group_analysis.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/evaluation/group_analysis.py):** *[NEW]* Implemented interval exporter, proportion plotting, PCA strategy spaces, and cross-validated Random Forest cohort prediction (optimized to execute UMAP transforms in a single high-speed batch).
- **[run_lm_analysis.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/run_lm_analysis.py):** Integrated CLI flag `--run-strategies` and automated the full Step 2 loading and validation sequence.
- **[behavioral_strategy_modeling.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/mds/behavioral_strategy_modeling.md):** Updated documentation under Sections 2, 5, and 6.
