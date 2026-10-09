# Behavioral Strategy Modeling at Macroscopic Timescales (~128s)

This document serves as a reference manual for defining and validating higher-order, slow-varying **behavioral strategies (the "vibe")** in mouse open-field exploration. It captures the conceptual paradigms, comparative modeling frameworks, fusion methods, and biological validation protocols synthesized for the MoSeq + Fiber Photometry pipeline.

---

## 1. Paradigm: The Macro vs. Micro Division

Standard ethological analyses focus on continuous posture trajectories or segmented discrete syllables (Tactical: ~0.5s–1.0s). This framework seeks to capture macroscopic intent (Strategic: ~128s) by analyzing sliding windows over sequences of embeddable tokens containing continuous micro-kinematic deviations.

```
       Tactical Level (~0.5s - 1s)                 Strategic Level (~128s / 2 min)
  ┌─────────────────────────────────────┐      ┌─────────────────────────────────────┐
  │ Syllables (MoSeq AR-HMM States)     │ ───> │ Strategies ("Vibes" / Intent)       │
  │ - ID: discrete token (e.g. 14, 8)   │      │ - Slow-varying state mixture        │
  │ - Duration: variable bout length    │      │ - Empirically optimized window size │
  │ - Kinematics: speed, posture var    │      │ - Biological validation (DA shifts) │
  └─────────────────────────────────────┘      └─────────────────────────────────────┘
```

---

## 2. Comparative Analysis of Macro-Level Frameworks

Five primary mathematical fields offer distinct ways to model sliding window token distributions to extract the Strategy ID.

| Framework | Novelty in Ethology | Implementation Feasibility | Compute Complexity | Interpretability (Mapping to DA) | Major Pros | Major Cons |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Music Info Retrieval (MIR) / Spectrograms** | **High** (Treats behavior as acoustic textures) | **High** (Standard SciPy DSP or `librosa`) | **Low-Medium** (Fourier transforms, fast runtime) | **High** (Intuitive concepts like behavioral tempo, rhythm, crest factors) | Fast to prototype; captures continuous instance-level micro-kinematics. | Struggles to elegantly model purely categorical, non-rhythmic syntax transitions. |
| **Dynamic Topic Modeling (NLP / LDA)** | **Medium** (Topic models known; transformer token fusion is novel) | **Very High** (Built-in `scikit-learn` LDA, `gensim`, or `BERTopic`) | **Medium** (Relies on pre-computed embeddings and standard HDBSCAN) | **High** (Strategies explicitly defined by mixtures of dominant syllables) | Extremely fast to spin up; natively handles sliding document windows (128s). | Discards instantaneous sequential syntax (treats windows as "bags of syllables"). |
| **Hierarchical HMMs (HDP-HMM)** | **Low-Medium** (HMMs are standard; macro-scale is new) | **High** (Can build on existing frameworks like `ssm` package) | **High** (MCMC sampling scales poorly on massive datasets) | **Very High** (Explicit transition matrices show grammatical shifts) | Highly rigorous; inherently probabilistic; provides explicit state boundaries. | Mathematically rigid; hard to integrate continuous micro-kinematics without parameter explosion. |
| **Time-Contrastive Networks (TCN)** | **Medium-High** (Growing in vision/neuro, novel for token streams) | **Medium** (Requires PyTorch/JAX to define positive/negative temporal pairs) | **High** (Deep learning; requires hyperparameter tuning) | **Low** (Outputs abstract black-box latent vectors; needs UMAP/SHAP) | End-to-end learning; completely bypasses manual feature engineering. | Risk of overfitting to session-specific recording noise over true behavior. |
| **Topological Data Analysis (TDA)** | **Very High** (Almost entirely unexploited for macro behavior) | **Low-Medium** (Requires topological libraries like `Gudhi` or `Ripser`) | **Very High** (Combinatorial homology scales poorly with data size) | **Medium** (Intuitive shape concepts: behavioral loops vs. linear branches) | Immune to kinematic noise; focuses on structural shape and invariant topology. | Highly sensitive to preprocessing; severe computational bottlenecks on long files. |

### Implementation Detail: Custom BERTopic Pipeline
We implement a zero-dependency, memory-safe version of the **BERTopic** pipeline tailored for behavioral tokens:
1. **Document Representation**: Frame-aligned rich tokens are grouped into 128s sliding windows (10s stride) to form behavioral "documents". Individual rich tokens are mapped to semantic strings (e.g., `"4_slow"`, `"53_fast"`) based on velocity ranks.
2. **CountVectorizer**: Reconstructs term-frequency counts of these token strings.
3. **UMAP**: Reduces the sparse term-frequency representations into a dense 5-dimensional strategy space.
4. **HDBSCAN**: Identifies high-density strategy clusters. Outlier windows assigned to cluster `-1` are resolved by projecting them to the nearest valid cluster centroid in UMAP space to guarantee continuous strategy sequences.

---

## 3. Fusing Discrete & Continuous Features (The Micro-Level)

Before a macro-framework can model a 128s window, the three core data streams for each **syllable bout** must be fused:
1. **Discrete ID:** Categorical syllable token (from MoSeq).
2. **Variable Duration:** Temporal length of the individual bout.
3. **Micro-Kinematics:** Postural trajectories, speed, and acceleration profiles.

### A. Deep Learning Fusion Architectures
*   **Multi-Modal Transformers (Cross-Attention Fusion):** Pass discrete embeddings through a sequence encoder, project continuous kinematics through a 1D convolution layer, and fuse them via cross-attention.
*   **VQ-VAE ("Rich Tokens"):** Encode the ID, duration, and continuous kinematics of a single bout into a low-dimensional space and quantize it. This generates a new discrete vocabulary (e.g. `Syllable 14-Fast-Short`).
*   **Temporal Graph Networks (TGN):** Build a graph of the 128s window where nodes are unique syllables, edges represent transition rates/durations, and node features are concatenated kinematics.

### B. Practical Baseline: Kinematic Sub-State Clustering (No-GPU / Low-Overhead)
To prototype without deep learning, we cluster continuous features *locally* per syllable:
1. For each syllable ID (e.g., Syllable 14), compile all occurrences across all sessions.
2. Form a feature vector for each bout: $\vec{z} = [\text{duration}, \text{mean\_velocity}, \text{mean\_angular\_velocity}, \text{mean\_acceleration}, \text{mean\_dist\_from\_center}]$.
3. Fit a lightweight **K-Means** ($k=3$) on these vectors.
4. Replace the generic sequence ID `14` with the resulting sub-state ID (e.g., `14_slow_short`, `14_fast_long`).
5. Feed these new enriched tokens directly into the Dynamic Topic Model (LDA) or HDP-HMM.

**Critical preprocessing — log1p transform:** Kinematic features are heavily right-skewed
(duration skew ≈ 43, angular velocity skew ≈ 18). K-Means assumes Gaussian-like distributions
and will otherwise place one centroid to chase a sparse tail of extreme bouts, leaving that
substate with < 1% of bouts — a "ghost token" that is effectively invisible to downstream
topic models. To correct this, a `log1p` transform is applied to all rate-based features
(duration, velocity, angular velocity, acceleration) **before** `StandardScaler`:

```python
_LOG1P_COLS = [0, 1, 2, 3]  # duration, velocity, angular_velocity, acceleration
X[:, _LOG1P_COLS] = np.log1p(X[:, _LOG1P_COLS])
# then StandardScaler → KMeans
```

`log1p` is safe for zero-valued bouts and compresses the long tail to make distributions
near-Gaussian. This reduced severely imbalanced syllables from 60/95 → 18/95 across the
full dataset (79,604 bouts), and the max/min cluster ratio from 117× to 8.6×.
`mean_dist_from_center` is left untransformed (skew = −0.40, already near-Gaussian).
See `mds/updates/tuesday.md` for the full empirical comparison.

---

## 4. End-to-End Pipeline Combinations

The micro-fusion methods plug directly into the macro-frameworks:

```
               [ Bout Micro-Features ]
             (ID, Duration, Kinematics)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │       FUSION ENGINE             │
        │ - VQ-VAE or K-Means/GMM         │
        └─────────────────────────────────┘
                         │
                         ▼ (Enriched Tokens)
        ┌─────────────────────────────────┐
        │       MACRO FRAMEWORK           │
        │ - LDA, HDP-HMM, or Spectrogram  │
        └─────────────────────────────────┘
                         │
                         ▼ (128s Strategy Vectors)
        ┌─────────────────────────────────┐
        │       CLUSTERING ENGINE         │
        │ - UMAP + GMM or HDBSCAN         │
        └─────────────────────────────────┘
                         │
                         ▼
                   [ Strategy IDs ]
```

1. **VQ-VAE / K-Means $\to$ HDP-HMM ("Rich Syntax"):** Reduces high-dimensional micro-kinematics to rich discrete tokens, keeping HMM parameter spaces small while preserving physical details.
2. **Multi-Modal Transformer $\to$ BERTopic ("Semantic Vibe"):** Leverages contextualized bout representations in document topic modeling. The most interpretable and computationally feasible deep-learning route.
3. **TGN $\to$ TDA ("Geometric Shape"):** Maps transitions to graphs and computes persistent homology to identify topological loops (e.g. repetitive grooming) vs. linear pathways (exploration).

## 5. Cohort and Genotype Validation

To ensure discovered strategies are biologically anchored and ethologically meaningful, we validate them using cohort prediction and similarity metrics:

### A. Experimental Group Classification (Predictive Superiority)
*   **Concept:** If macroscopic strategies represent true behavioral states, session-level strategy proportions should predict experimental group labels (e.g. Saline vs. Amphetamine, WT vs. KI) more compactly and accurately than raw MoSeq syllables.
*   **Test:** Train a Random Forest Classifier using Stratified K-Fold cross-validation on:
    - Base MoSeq Syllables (100 features)
    - Rich Tokens (294 features)
    - BERTopic Strategy proportions (15 features)
*   **Result:** Strategies achieve equal or superior accuracy (e.g., **0.5333**) to raw syllables (**0.5067**) and rich tokens (**0.5200**), compressing features by **7-20×** and validating that strategy mixtures capture cohort distinctions.

### B. Cohort Proportions & Similarity Space
*   **Concept:** Strategy proportions should be consistent within cohorts and show clear segregation between drug/genotype groups.
*   **Test:** Plot average strategy proportions by group and map sessions into a 2D PCA space derived from strategy proportions.
*   **Result:** Visual clustering of cohorts in the PCA space confirms that macroscopic strategy profiles are robust indicators of pharmacological (Amphetamine) and genetic (KI vs. WT) groups.

---

## 6. Implementation Progress (SWE Build Order)

All core baseline and cohort validation components have been implemented:
*   **[x] Phase 1: Plumbing:** Slices raw frame sequences into sliding windows of 128s (segmentation.py).
*   **[x] Phase 2: Custom BERTopic Pipeline:** Implemented CountVectorizer + UMAP + HDBSCAN with dynamic semantic labeling and outlier resolution (strategies.py).
*   **[x] Phase 3: Cohort Validation & Export:** Strategy interval exports to CSV, cohort proportion plots, and Random Forest predictive validation (group_analysis.py).
