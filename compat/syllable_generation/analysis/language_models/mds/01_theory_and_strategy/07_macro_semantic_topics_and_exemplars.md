# 07: Macro-Semantic Topics, Exemplar Retrieval, and Strategic Manifolds

---

## Executive Summary

While **Modules 01–06** establish the mathematical foundations of sub-second generative syllabization, non-Markovian memory curves, sequence architectures, dopamine coupling, and statistical phenotyping, this document details the **macro-semantic translation engine**.

A common challenge in quantitative ethology is that:
$$\boxed{\text{A syllable is interpretable, but 10,000 discrete syllables are not an interpretation of behavior.}}$$

To bridge sub-second syllables to digestible biological regimes, this module operationalizes **Behavioral Semantic Manifolds**, **The Continuum Principle (Strategy $\neq$ Cluster)**, **Exemplar Retrieval**, and **Macro-State Grammars** on our empirical 30-session open-field dataset ($T = 174,686\text{ bouts}$), providing actionable protocols that do not require external foundation model pretraining.

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
│                                  THE 4-TIER BEHAVIORAL HIERARCHY                                  │
+───────────────────────────+───────────────────────────+───────────────────────────────────────────+
│ Tier 1: Action            │ Tier 2: Episode           │ Tier 3: Strategy / Topic                  │
│ (~400 ms)                 │ (~68.3 s Window)          │ (Macroscopic Semantic Space)              │
├───────────────────────────┼───────────────────────────┼───────────────────────────────────────────┤
│ Syllables: s_t in {1..K}  │ Behavioral Document W_t   │ Continuous Vector: z_t in R^d (d=8..16)   │
│ (e.g. Rear, Sniff, Run)   │ Feature vector z_t in R^d │ Continuous Gradients, Attractors & Modes  │
+───────────────────────────+───────────────────────────+───────────────────────────────────────────+
                                                        │
                                                        ▼
+───────────────────────────────────────────────────────────────────────────────────────────────────+
│                             Tier 4: Phenotype & Neural Dynamics                                   │
├───────────────────────────────────────────────────────┬───────────────────────────────────────────┤
│ • Cohort Contrast: WT vs. KO / PFF vs. Monomer        │ • Photometry Coupling: DA & ACh Dynamics  │
+───────────────────────────────────────────────────────┴───────────────────────────────────────────+
```

---

## 1. The Behavioral Document Abstraction ($W_t \to \mathbf{z}_t$)

We define each time-anchored window of duration $W = 68.3\text{ s}$ (spanning approximately $N \approx 60\text{–}120$ discrete bouts) as a **Behavioral Document** $W_t = (s_{t-W}, \dots, s_t)$.

To evaluate what information constitutes a macro-strategy, we construct **four competing mathematical representations** $\mathbf{z}_t$ for the exact same physical window:

```
                               68.3-SECOND BEHAVIORAL EPISODE (W_t)
                                                │
       ┌────────────────────────┬───────────────┴───────────────┬────────────────────────┐
       ▼                        ▼                               ▼                        ▼
  [ Representation A ]     [ Representation B ]            [ Representation C ]     [ Representation D ]
     Bag-of-Words            Count + Duration                Action2Vec Pooling        Contextual LM
  u_count in R^100         [u_count, u_dur] in R^200       Mean/Max e_s in R^32     z_t from Transformer
  (Pure Repertoire)        (+ Pacing & Intensity)          (Distributional Semantics)  (Full Sequential Syntax)
```

### A. Representation A: Bag-of-Words Count Vector ($\mathbf{u}_{\text{count}} \in \mathbb{R}^K$)
* **Definition:** Normalized frequency of each syllable token within window $W_t$:
  $$u_{\text{count}, k} = \frac{1}{|W_t|} \sum_{i \in W_t} \mathbb{I}(s_i = k)$$
* **Ethological Meaning:** Pure behavioral repertoire allocation (action budget).
* **Interpretability:** Maximum. Directly reveals which behaviors are over- or under-represented.

### B. Representation B: Joint Count + Duration Vector ($[\mathbf{u}_{\text{count}}, \mathbf{u}_{\text{dur}}] \in \mathbb{R}^{2K}$)
* **Definition:** Syllable count frequencies concatenated with cumulative dwell times:
  $$u_{\text{dur}, k} = \frac{1}{\sum_{i \in W_t} d_i} \sum_{i \in W_t} d_i \cdot \mathbb{I}(s_i = k)$$
* **Ethological Meaning:** Captures both action occurrence and **behavioral persistence/intensity** (e.g., distinguishing 10 brief, fragmented rears from 2 sustained, continuous rears).

### C. Representation C: Distributional Action2Vec Pooling ($\mathbf{z}_{\text{A2V}} \in \mathbb{R}^p$)
* **Definition:** Syllable embeddings $\mathbf{e}_k \in \mathbb{R}^{32}$ trained via Skip-Gram with Negative Sampling (SGNS, context $c=3$ bouts) on the 30-session corpus, pooled across the window:
  $$\mathbf{z}_{\text{A2V}} = \frac{1}{|W_t|} \sum_{i \in W_t} \mathbf{e}_{s_i}$$
* **Ethological Meaning:** Syllables that occur in interchangeable behavioral contexts share similar embedding vectors ($\mathbf{e}_{\text{sniff\_wall}} \approx \mathbf{e}_{\text{rear\_wall}}$). Pooling them captures the continuous semantic neighborhood of the episode.

### D. Representation D: Contextual Sequence Embedding ($\mathbf{z}_{\text{ctx}} \in \mathbb{R}^{d_{\text{model}}}$)
* **Definition:** The final hidden state or mean-pooled activation vector from a Causal Transformer / Mamba encoder trained on syllable sequences:
  $$\mathbf{z}_{\text{ctx}} = \text{Encoder}(s_{t-W:t}, \mathbf{d}_{t-W:t})$$
* **Ethological Meaning:** Context-dependent representation where the embedding of a syllable is altered by its chronological history (e.g., *sniffing after locomotion* vs. *sniffing after freezing*).

---

## 2. The Continuum Principle: Why Strategy $\neq$ Discrete Cluster

A critical conceptual hazard in computational ethology is **the clustering fallacy**—the assumption that every biological state must partition into discrete, isolated boxes ($k \in \{1, \dots, 16\}$). 

In reality:
$$\mathbf{z}_t \in \mathbb{R}^d \quad (d \approx 8\text{–}16)$$
where animal behavior frequently manifests as a **smooth continuum** rather than disjoint clusters:

$$\boxed{\text{Exploratory Foraging} \quad \longleftrightarrow \quad \text{Cautious Investigation} \quad \longleftrightarrow \quad \text{Stationary Vigilance}}$$

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           THE BEHAVIORAL MANIFOLD TOPOLOGY SPECTRUM                             │
├───────────────────────────────┬───────────────────────────────┬─────────────────────────────────┤
│ 1. Continuous Gradients       │ 2. Attractor Basins           │ 3. Trajectory Vector Fields     │
│ Smooth transitions along      │ Potential energy wells where  │ Dynamic flow fields:            │
│ continuous behavioral axes    │ trajectories dwell and linger │ v(z) = E[ dz/dt | z ]           │
├───────────────────────────────┼───────────────────────────────┼─────────────────────────────────┤
│ 4. Metastable Saddles         │ 5. Truly Discrete Modes       │ 6. Recurring Macro-Episodes     │
│ Choice-points & branching     │ Isolated islands (e.g. rigid  │ Closed periodic orbits in       │
│ decision boundaries           │ cephalocaudal grooming)       │ latent space                    │
└───────────────────────────────┴───────────────────────────────┴─────────────────────────────────┘
```

### Mathematical Formulation of the Continuous Semantic Space:
Rather than forcing all evaluation windows into categorical assignments, we define the **Behavioral Semantic Space** as a Riemannian manifold $(\mathcal{M}, g)$ with coordinates $\mathbf{z}_t \in \mathbb{R}^d$ ($d = 8\text{–}16$) and characterize its topological properties:

1. **Continuous Axis Gradient $\nabla \Phi(\mathbf{z})$:**
   * Identifying principal semantic gradients (e.g., *Kinetic Arousal Axis* $\mathbf{v}_1$, *Environmental Risk/Thigmotaxis Axis* $\mathbf{v}_2$).
2. **Empirical Potential Energy Landscape $U(\mathbf{z})$:**
   * Defined from the stationary probability density $p(\mathbf{z})$:
     $$U(\mathbf{z}) = -\ln p(\mathbf{z})$$
   * **Attractors (Valleys):** Local minima of $U(\mathbf{z})$ represent stable behavioral regimes where mice linger.
   * **Saddle Points (Ridges):** Transition barriers between distinct modes.
3. **Manifold Drift Vector Field $\mathbf{F}(\mathbf{z})$:**
   * The instantaneous expected velocity across strategy space:
     $$\mathbf{F}(\mathbf{z}) = \mathbb{E}\left[ \frac{\mathbf{z}_{t+\Delta} - \mathbf{z}_t}{\Delta} \;\middle|\; \mathbf{z}_t = \mathbf{z} \right]$$

---

## 3. Topic Modeling & Manifold Decomposition Frameworks

To capture both continuous gradients and emergent discrete modes, we evaluate four complementary mathematical decompositions:

```
Episode Matrix: Z in R^{N x d} ──► [ Decomposition Engine ] ──► Strategy Manifold / Topics
                                  • Truncated SVD (Linear Eigenspace & Gradients)
                                  • LDA (Soft Dirichlet Mixtures)
                                  • NMF (Non-negative Part-based)
                                  • UMAP + HDBSCAN (Density-Based Topological Clusters)
```

### A. Continuous Strategy Eigenspace (SVD / PCA)
* Truncating to $d = 8\text{–}16$ principal components captures $80\text{–}90\%$ of total strategic variance across sessions without imposing artificial cluster boundaries.
* Preserves continuous Euclidean and geodesic distances across the full behavioral spectrum.

### B. Latent Dirichlet Allocation (LDA) / NMF (Soft Mixtures)
* Treating the episode-by-syllable matrix $\mathbf{U}_{N \times K}$ as a document-term matrix:
  $$\boldsymbol{\theta}_t = [\theta_{t, 1}, \theta_{t, 2}, \dots, \theta_{t, M}] \in \Delta^M, \quad \sum_{m=1}^M \theta_{t, m} = 1$$
* **Biological Value:** Yields soft, mixed strategy memberships rather than forcing hard discrete assignments:
  $$\text{Episode } W_t = 60\%\text{ Active Perimeter Exploration} + 25\%\text{ Vigilance} + 15\%\text{ Grooming}$$

### C. Multi-Method Convergence Validation
A candidate state or cluster is considered an **invariant biological attractor** if and only if it is independently discovered across distinct mathematical formalisms:
$$\text{State Robustness Score } \mathcal{R}(S) = \text{Adjusted Rand Index}\left( \mathcal{C}_{\text{LDA}}, \mathcal{C}_{\text{NMF}}, \mathcal{C}_{\text{SVD}}, \mathcal{C}_{\text{HDBSCAN}} \right)$$

---

## 4. Interpretable Ethological Topic Cards (Exemplar Retrieval)

Rather than presenting abstract mathematical centroids ($\boldsymbol{\mu}_k \in \mathbb{R}^{48}$), every discovered strategy region or attractor basin $k$ is automatically converted into an **Ethological Topic Card** backed by real animal video episodes:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TOPIC CARD: REGIME 04 — "CAUTIOUS PERIMETER INVESTIGATION"                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Manifold Geometry: Continuous Attractor Basin (Depth: Delta U = 2.4 k_B T)                    │
│ • Global Occupancy: 18.4% of total dataset time budget (Rank 2/8)                               │
│ • Mean Regime Dwell Time: 38.2 ± 6.4 seconds (Log-normal distribution)                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TOP ENRICHED SYLLABLES (vs. Baseline Unigram Prior):                                            │
│   1. Syllable 14 (Supported Rear):      +142% enrichment (p < 1e-6)                             │
│   2. Syllable 18 (Wall-Sniff Sweep):    +118% enrichment (p < 1e-6)                             │
│   3. Syllable 31 (Micro-Pause Sniff):   +84%  enrichment (p < 1e-4)                             │
│   4. Syllable 08 (High-Speed Run):      -76%  depleted   (p < 1e-5)                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ SPATIAL ARENA GEOMETRY:                                                                         │
│   • Perimeter Wall Proximity: 88.4% of time in outer 10% perimeter (Thigmotaxis)                │
│   • Center Excursion Rate: 0.04 entries/min                                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MACRO-TRANSITION DYNAMICS:                                                                      │
│   • Most Likely Inflow Gradient:  Exploration -> Regime 04 (P = 0.44)                           │
│   • Most Likely Outflow Gradient: Regime 04 -> Stereotyped Grooming (P = 0.38)                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ REAL ANIMAL EXEMPLAR EPISODES:                                                                  │
│   1. Session `2024-03-12_WT_M2`, Episode Window `t = 340s - 408s` (Attractor Distance = 0.12)   │
│   2. Session `2024-03-15_WT_M5`, Episode Window `t = 820s - 888s` (Attractor Distance = 0.14)   │
│   3. Session `2024-04-02_KO_M1`, Episode Window `t = 120s - 188s` (Attractor Distance = 0.16)   │
│   [Clickable Link to Diagnostic Video Clip Generator]                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Retrieval Protocol:
1. Identify reference coordinate $\mathbf{c}_k$ (attractor minimum or topic peak) in the continuous manifold $\mathcal{Z}$.
2. Query the $K=10$ nearest historical evaluation windows $W_i$ minimizing Euclidean/geodesic distance $\|\mathbf{z}_i - \mathbf{c}_k\|_2$.
3. Automatically render multi-panel diagnostic videos combining synchronized overhead keypoints, raw video clips, syllable track bars, and real-time photometry traces.

---

## 5. The "Same Repertoire, Different Syntax" Validation Test

A central scientific question is whether **temporal syntax** carries distinct strategic information beyond the static syllable repertoire. We formalize the **Syntactic Disambiguation Test**:

```
                       REPERTOIRE-IDENTICAL PAIR IDENTIFICATION
                                           
        Episode A (Chronological Syntax)               Episode B (Permuted Syntax)
  [ Run -> Sniff -> Rear -> Pause -> Groom ]     [ Groom -> Pause -> Rear -> Sniff -> Run ]
  ──────────────────────────────────────────     ──────────────────────────────────────────
           u_count(A)  ≈  u_count(B)                     u_count(A)  ≈  u_count(B)
                  │                                              │
                  ▼                                              ▼
           z_ctx(A) in R^64                               z_ctx(B) in R^64
                  │                                              │
                  └──────────────► d(z_A, z_B) >> 0 ◄────────────┘
                                         │
                                         ▼
            D_JS( P_future(A) || P_future(B) ) >> 0  (DIVERGENT FUTURES)
```

### Test Formulation:
1. **Pair Mining:** Search all pairs of 68.3s evaluation windows $(W_A, W_B)$ across the dataset satisfying:
   $$\|\mathbf{u}_{\text{count}}(A) - \mathbf{u}_{\text{count}}(B)\|_2 \le \epsilon \quad (\epsilon \to 0)$$
2. **Contextual Divergence:** Compute the distance in contextual embedding space:
   $$d_{\text{ctx}}(A, B) = \|\mathbf{z}_{\text{ctx}}(A) - \mathbf{z}_{\text{ctx}}(B)\|_2$$
3. **Future Trajectory Divergence:** Measure the Jensen-Shannon Divergence between their ground-truth future syllable distributions over lookahead horizon $H = 120\text{ s}$:
   $$D_{\text{JS}}\left( P_{\text{future}}(A) \,\|\, P_{\text{future}}(B) \right)$$
4. **Hypothesis:** If $D_{\text{JS}} > 0$ with statistical significance ($p < 0.001$, permutation test), it formally proves that **identical syllable repertoires instantiate distinct strategic futures solely due to chronological syntax**.

---

## 6. Macro-Strategy Trajectories & Manifold Flow Fields

Just as sub-second syllables follow an action grammar (Level 1 $\to$ Level 2), continuous macro-strategies trace continuous trajectories $\mathbf{z}(t)$ through the semantic manifold:

```
                    CONTINUOUS MANIFOLD FLOW & TRAJECTORIES
                    
               [ Active Exploration Regime ]
                       │              ▲
                       │ v_flow       │ v_return
                       ▼              │
               [ Cautious Investigation ]
                       │
                       │ v_transition
                       ▼
               [ Stereotyped Grooming Basin ]
```

### Quantitative Metrics:
1. **Continuous Velocity Vectors $\Delta \mathbf{z}_t = \mathbf{z}_{t+\Delta} - \mathbf{z}_t$:** Measures the rate and direction of travel across the behavioral manifold.
2. **Trajectory Curvature & Persistence:**
   $$\kappa(t) = \frac{\|\dot{\mathbf{z}} \times \ddot{\mathbf{z}}\|}{\|\dot{\mathbf{z}}\|^3}, \quad \tau_{\text{persistence}} = \text{Autocorrelation Time of } \mathbf{z}(t)$$
3. **Macro-Transition Matrix $\mathbf{T}_{\text{macro}} \in \mathbb{R}^{M \times M}$ (Discretized Approximation):**
   $$T_{ij} = P(Z_{t + \Delta} = j \mid Z_t = i), \quad \Delta = 30\text{–}60\text{ seconds}$$

---

## 7. Coupling Macro-Strategies to Fiber Photometry (DA & ACh)

With continuous manifold coordinates $\mathbf{z}_t \in \mathbb{R}^d$ established, we formulate three testable neuroscience hypotheses connecting behavior to real-time **dLight1 (DA)** and **ACh3.0** photometry:

```
                            THREE MODES OF NEURAL COUPLING
                            
  1. Tonic State Encoding        2. Steering Vectors (Trajectories)     3. Strategic Surprise
  ───────────────────────        ──────────────────────────────────     ─────────────────────
     DA(t) ~ f( z_t )               DA(t) ~ f( z_{t+Delta} - z_t )         DA(t) ~ -log P(z'|z)
  (Dopamine tone reflects        (Dopamine bursts steer direction       (Dopamine signals unexpected
   position on the manifold)      of travel across semantic space)       manifold transitions)
```

### Protocol for Empirical Validation:
1. **Manifold Position Regression:** Evaluate correlation between tonic dopamine and continuous semantic axes $\mathbf{z}_t$:
   $$\text{dLight1}(t) = \mathbf{w}^T \mathbf{z}_t + b + \epsilon_t$$
2. **Directional Steering Regression:** Regress dopamine burst amplitude against the displacement vector $\Delta \mathbf{z}_t = \mathbf{z}_{t+30\text{s}} - \mathbf{z}_t$:
   $$\text{dLight1}(t) = \mathbf{v}^T \Delta \mathbf{z}_t + \epsilon_t$$
3. **Transition Hazard & Surprise:** Evaluate correlation between phasic dopamine bursts and transition information content:
   $$I_{\text{surprise}}(t) = -\log_2 P(\mathbf{z}_{t+\Delta} \mid \mathbf{z}_t)$$

---

## 8. Actionable Implementation Pipeline (Step-by-Step)

```
[ Step 1: Extract Episodes ] ──► Compute 4 representations (A, B, C, D) across all 30 sessions.
              │
[ Step 2: Fit Manifold & Topics ] ──► Run SVD (d=8..16), LDA (M=8), NMF (M=8), and evaluate potential energy U(z).
              │
[ Step 3: Exemplar Cards ]   ──► Extract top 10 real video episodes for key attractors & gradients.
              │
[ Step 4: Syntax Test ]      ──► Execute "Same Repertoire, Different Syntax" pair divergence test.
              │
[ Step 5: Photometry Map ]   ──► Regress dLight1/ACh3.0 timeseries against z_t and displacement vectors Delta z_t.
```
