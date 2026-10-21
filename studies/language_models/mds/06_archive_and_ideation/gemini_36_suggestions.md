# Architectural Review & Future Suggestions: Behavioral Language Modeling

**Author**: Gemini 3.6  
**Date**: July 22, 2026  
**Target Module**: `syllable_generation/analysis/language_models`  

---

## 1. Evaluation of the 2-Stage Strategy

The core framework relies on a **two-stage decoupled architecture**:
1. **Micro-Level Feature Engineering**: Extracting bout-level continuous kinematics (duration, velocity, acceleration, distance from center), applying `log1p` transformation and Z-score scaling, and clustering into $k$ kinematic sub-states ("rich tokens" e.g., `Syllable 14_fast`).
2. **Macro-Level Strategy Conversion**: Grouping rich tokens into sliding temporal windows ($\sim 60\text{s}$), vectorizing token streams (Count, Transition, or Soft-Count matrices), projecting via UMAP, and discovering macroscopic strategies ("vibes") using HDBSCAN or GMMs.

### Why This Design is Highly Effective
- **Decoupled Abstractions**: Separating bout-level physical sub-states from macroscopic window strategies allows independent hyperparameter tuning, debugging, and validation.
- **Ethological Interpretability**: Unlike end-to-end black-box deep neural networks, every strategy cluster can be explicitly decomposed into its constituent micro-behaviors and kinematic sub-states.
- **Compute & Memory Efficiency**: Quantizing high-frequency continuous data into discrete rich token vocabularies allows macro-level topic models (BERTopic, UMAP, HDBSCAN) to run in seconds with low memory overhead.

---

## 2. Four Key Technical & Architectural Recommendations

### Recommendation 1: $N$-Gram Syllable Syntax Vectorization (Beyond 1-Step Transitions)
- **Observation**: In current hyperparameter sweeps, 1-step transition probability matrices ($P(S_{t+1} \mid S_t)$) systematically outperform simple term counts, achieving top decodability accuracies (**65.00%**).
- **Proposal**: Extend the vectorization module ([strategies.py](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/lib/models/strategies.py)) to include:
  1. **$N$-gram motif frequencies** (e.g., 2-gram or 3-gram syllable sequences like `Rear -> Scan -> Groom`).
  2. **Decayed 2-step transition matrices** $P(S_{t+2} \mid S_t)$.
- **Impact**: Captures multi-syllable behavioral syntax motifs inside sliding windows without parameter explosion.

---

### Recommendation 2: Soft Membership Vectors for HDBSCAN Outlier Resolution
- **Observation**: Outliers labeled `-1` by HDBSCAN are currently hard-assigned to the nearest cluster centroid in UMAP space using Euclidean distance.
- **Proposal**: Utilize HDBSCAN’s soft-clustering membership vectors (`hdbscan.all_points_membership_vectors(hdbscan_model)`). Represent each window as a continuous probability distribution $\vec{p} = [p_1, p_2, \dots, p_K]$ across strategies.
- **Impact**: Behavioral strategy shifts in open-field exploration are continuous and probabilistic. Soft-assignment eliminates artificial state flickering at strategy transition boundaries.

---

### Recommendation 3: Neural Correlate Validation via Synchronized Photometry
- **Observation**: Model validation currently relies on Random Forest classification accuracy (WT vs. KI or Saline vs. Amphetamine).
- **Proposal**: Implement **Photometry Cross-Correlation Alignment** (as outlined in [sliding_window_optimization.md](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/mds/sliding_window_optimization.md)). Compute Mutual Information or Event-Triggered Averages (ETA) of dLight (dopamine) signals aligned to strategy transition boundaries ($S_t \neq S_{t-1}$).
- **Impact**: Demonstrating that macro-strategy switches correlate with tonic dopamine ramps or transient DA drops provides undeniable neurobiological validation for the discovered strategy clusters.

---

### Recommendation 4: Gaussian Process Bayesian Optimization for Window Tuning
- **Observation**: Manual grid sweeps over window duration ($W$), step size ($S$), and substate count ($k$) require hundreds of evaluations.
- **Proposal**: Implement Gaussian Process Bayesian Optimization (GPR with Matérn 2.5 kernel and Expected Improvement acquisition function) to jointly optimize $W \in [10\text{s}, 128\text{s}]$ and $k \in [2, 6]$.
- **Impact**: Discovers global peak KL information gain and cohort decodability in ~30–50 evaluations instead of manual grid searches.

---

## 3. Extended Hyperparameter Search Space Design (Directly Comparable Sweep)

To evaluate these new architectural extensions against current baseline runs without introducing confounding variables, we design an **Extended Factorial Search Space**.

### A. Design Philosophy for Fair Comparison
1. **Control Anchor**: Every parameter combination in the current top-performing sweep (`feature_mode="postural"`, `cluster_model="gmm"`, `vectorizer="transition"`) is included as an explicit control baseline.
2. **Normalized Overlap Ratio**: Window duration ($W$) and stride ($S$) are paired to enforce a constant **75% overlap ratio** ($S = W / 4$) across all window scales.
3. **Additive Factorial Structure**: New dimensions ($N$-gram vectorization, soft outlier resolution, window scales) are added orthogonality to quantify the exact $\Delta \text{Accuracy}$ and $\Delta F_1$ provided by each improvement.

---

### B. Extended Factorial Search Grid

| Factor | Extended Options | Control Baseline | Description |
|:---|:---|:---:|:---|
| **1. Vectorizer Type** | `transition`, `transition_2step`, `ngram_2gram`, `ngram_3gram`, `soft_count` | `transition` | Tests 1-step vs. 2-step decayed transitions vs. 2-gram/3-gram phrase frequencies. |
| **2. Substate Count ($k$)** | `[2, 3, 4, 5]` | `3` | Tests substate granularity (2 to 5 sub-states per syllable). |
| **3. Outlier Mode** | `hard_centroid`, `soft_hdbscan` | `hard_centroid` | Tests hard Euclidean UMAP centroid mapping vs. continuous HDBSCAN soft-assignment. |
| **4. Temporal Window ($W, S$)** | `(30s, 7.5s)`, `(60s, 15s)`, `(90s, 22.5s)` | `(60s, 15s)` | Tests micro-macro temporal window scaling at a fixed 75% overlap ratio. |
| **5. UMAP $n_{\text{neighbors}}$** | `[30, 45]` | `30` | Controls manifold local vs. global structure preservation. |
| **6. HDBSCAN $min\_cluster\_size$** | `[50, 75, 100]` | `75` | Controls minimum density scale for strategy cluster discovery. |

---

### C. Search Space Complexity & Execution Strategy

- **Total Grid Size**: $5 \text{ Vectorizers} \times 4 \text{ Substates (k)} \times 2 \text{ Outlier Modes} \times 3 \text{ Window Scales} \times 2 \text{ Neighbors} \times 3 \text{ Cluster Sizes} = \mathbf{720 \text{ Configurations}}$.
- **Resumption & Caching**: Existing completed runs (e.g., `postural_gmm_k3_transition_n30_mcs75`) are automatically matched by filename and skipped.
- **Comparative Metrics Saved**:
  - `Classifier Accuracy` & `Weighted F1-Score` (WT vs. KI and Saline vs. AMPH decodability).
  - `Feature Dimension Count` (Feature efficiency ratio).
  - `Outlier Coverage` (% of frames assigned valid strategy IDs vs. noise).

---

## 4. Summary & Next Steps

1. **Maintain the 2-Stage Pipeline**: Keep the feature engineering and strategy conversion stages modular.
2. **Incorporate $N$-gram Syntax**: Add 2-gram/3-gram transition vectorization to `StrategyBERTopic`.
3. **Validate with Photometry**: Align dLight traces with strategy switch frames.
