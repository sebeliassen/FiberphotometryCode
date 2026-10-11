# Behavioral Strategy Modeling Pipeline - Status Update (Monday, August 3, 2026)

This document records the work completed on Monday, August 3, 2026, resuming work following the 2-week leave and establishing the **Strategy Scale & Entropy Optimization Framework**.

---

## 1. What We Did Today

### A. Context Audit & Run History Synthesis
* Audited past conversation transcripts and sweep artifacts across all 3 historical hyperparameter sweeps:
  1. **Run 1 (Broad Sweep, Jun 25)**: 88 configs (`kmeans` vs `gmm`, `kinematic` vs `postural`), Peak Acc: **65.00%** (`postural_gmm_transition`).
  2. **Run 2 (Limited Sweep, Jul 22-23)**: 108 configs (Sub-states $k \in 2..4$), Peak Acc: **68.33%** (`hybrid_gmm_k4_soft_count`).
  3. **Run 3 (Extended Sweep, Jul 23-24)**: 240 configs (2-step decayed transitions, bi-grams, tri-grams, $k \in 2..5$), Peak Acc: **65.00%**.

### B. Addressing the "Category Over-Granularity" Challenge
* Identified that discovering **34 strategy categories** across 100 base syllables risked creating micro-motifs rather than broad, interpretable macro-behavioral states ("vibes").
* Formulated **3 distinct mathematical approaches** to control strategy scale and added quantitative **Entropy Balance ($H_{\text{norm}}$)** and **Effective Cluster Count ($K_{\text{eff}}$)** metrics to detect collapsed or ghost cluster distributions.

### C. Execution of the 46-Run Comparative Sweep
* Implemented and executed [run_vibe_scale_sweep.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/run_vibe_scale_sweep.py), evaluating 46 configurations across:
  * **Approach A (2-Tier Hierarchy)**: Merging 34 base sub-strategies into $K \in [4, 6, 8, 10, 12]$ Super-Regimes via Cosine, Euclidean, and Average linkage.
  * **Approach B (HDBSCAN Scale Tuning)**: Varying $mcs \in [100, 150, 200, 250, 300, 400, 500]$ and $n\_neighbors \in [15, 30, 45]$.
  * **Approach C (Direct Fixed-$K$ Mixture Models)**: Window-level GMM and LDA directly for $K \in [4, 6, 8, 10, 12]$.

---

## 2. Key Findings & Results (How & Why)

### 🏆 Top Ranked Configurations

| Rank | Approach | Configuration Name | Active Categories ($K$) | Effective $K_{\text{eff}}$ | Entropy Balance ($H_{\text{norm}}$) | Classifier Accuracy |
|:---:|---|---|:---:|:---:|:---:|:---:|
| **1** | **Approach B (HDBSCAN Scale)** | `ApproachB_HDBSCAN_mcs150_n30` | **20** | **15.94** | **0.924** | **71.67%** 🏆 |
| **2** | **Approach B (HDBSCAN Scale)** | `ApproachB_HDBSCAN_mcs300_n30` | **9** | **7.97** | **0.944** | **66.67%** 🥈 |
| **3** | **Approach A (2-Tier Hierarchy)** | `ApproachA_Hierarchy_K12_cosine` | **12** | **6.86** | **0.775** | **65.00%** 🥉 |
| **4** | **Approach B (HDBSCAN Scale)** | `ApproachB_HDBSCAN_mcs250_n30` | **10** | **8.93** | **0.951** | **63.33%** |
| **5** | **Approach A (2-Tier Hierarchy)** | `ApproachA_Hierarchy_K10_euclidean` | **10** | **8.68** | **0.938** | **63.33%** |

### Key Architectural Takeaways

1. **New Project Peak Decodability (71.67%)**:
   * **`ApproachB_HDBSCAN_mcs150_n30`** achieved **71.67% accuracy** (an all-time project high across the 4 cohorts vs. **36.67%** raw syllable baseline).
   * Modest density regularization ($mcs=150$) pruned minor noise while retaining **20 active macro-strategies** with strong balance ($H_{\text{norm}} = 0.924$).

2. **The 9-Category "Sweet Spot" Model (`ApproachB_HDBSCAN_mcs300_n30`)**:
   * Achieved **66.67% accuracy** with **only 9 macro-categories** ($K_{\text{eff}} = 7.97$, $H_{\text{norm}} = 0.944$).
   * **Native Single-Pass Model**: Operates natively inside `StrategyBERTopic(n_neighbors=30, min_cluster_size=300)`. HDBSCAN's density tree automatically coalesces small sub-clusters into 9 dense macro-islands during a single `.fit()` pass without requiring post-hoc rules.

3. **Approach C Failure Mode**:
   * Direct GMM/LDA models without UMAP capped at **36.67% – 50.00% accuracy**.
   * LDA suffered severe entropy collapse ($H_{\text{norm}} \approx 0.45$), assigning 90%+ of window probability to 1–2 dominant states. Manifold projection via UMAP is strictly required.

---

## 3. Plan on How to Continue (Next Steps Roadmap)

```
                                    ROADMAP FOR NEXT STEPS

  +---------------------------------------------------------------------------------------------------+
  | STEP 1: Frame-Level Strategy Timeline Export                                                      |
  | Export session_strategy_intervals.csv for 9-vibe model (ApproachB_HDBSCAN_mcs300_n30)            |
  +---------------------------------------------------------------------------------------------------+
                                              │
                                              ▼
  +---------------------------------------------------------------------------------------------------+
  | STEP 2: Cohort Strategy Characterization & Presentation Plots                                      |
  | Generate strategy proportion bar plots, 2D PCA strategy spaces, and transition probability matrices|
  +---------------------------------------------------------------------------------------------------+
                                              │
                                              ▼
  +---------------------------------------------------------------------------------------------------+
  | STEP 3: Fiberphotometry Alignment (Biological Validation)                                         |
  | Compute dLight (dopamine) PETHs / Event-Triggered Averages aligned to strategy switch frames       |
  +---------------------------------------------------------------------------------------------------+
```

### Action Items for Next Session:
1. **Export 9-Vibe Strategy Intervals**:
   * Run-length encode (RLE) the frame-by-frame strategy assignments for all 60 AMPH sessions using `ApproachB_HDBSCAN_mcs300_n30` and save to `output/strategies_extended/session_strategy_intervals.csv`.
2. **Generate Cohort Visualizations for Short Presentation**:
   * Generate `strategy_distribution_by_group.png` (stacked bar chart comparing % time in 9 vibes across `WT_Saline`, `WT_AMPH`, `KI_Saline`, `KI_AMPH`).
   * Plot 2D PCA strategy space showing cohort clustering.
3. **Biological Validation (dLight PETH Alignment)**:
   * Align continuous photometry signals ($\Delta F/F$ for L-VS and R-DMS) to strategy switch boundaries ($S_t \neq S_{t-1}$) to test whether strategy shifts correlate with tonic dopamine ramps or transient surges.
