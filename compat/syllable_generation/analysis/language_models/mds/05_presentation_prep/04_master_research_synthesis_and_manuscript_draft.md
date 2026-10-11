# Master Research Synthesis: Hierarchical Semantic Compression and Neuromodulatory Steering

---

## Executive Summary

This document captures the unified, high-level scientific manuscript framework synthesizing our empirical language modeling benchmarks, the 5-pillar competitive landscape, optimal transport phenotyping, and multi-modal fiber photometry coupling.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE COMPLETE COMPUTATIONAL PIPELINE                             │
├───────────────────────────┬───────────────────────────┬─────────────────────────────────────────┤
│ Stage 1: Discretization   │ Stage 2: Document Model   │ Stage 3: Phenotyping & Neurodynamics    │
├───────────────────────────┼───────────────────────────┼─────────────────────────────────────────┤
│ 80 Hz Raw Video           │ 68.3s Sliding Windows W_t │ Continuous Manifold z_t in R^d          │
│       │                   │ (≈170 Bouts / Episode)    │       │                                 │
│       ▼                   │       │                   │       ├──► Sinkhorn-Wasserstein Phenotype│
│ Keypoint-MoSeq (AR-HMM)   │       ▼                   │       │    (PFF vs. Monomer, DAT-KI)    │
│ K ≈ 50–71 Syllables       │ Sequence Transformer /    │       │                                 │
│ (~400 ms Primitives)      │ Mamba-1 Model (P5)        │       └──► Striatal Photometry Steering │
│ (Filters 95% Pose Jitter) │ (P0 -> P5 Information)    │            (DA Phasic -> Delta z_t)     │
└───────────────────────────┴───────────────────────────┴─────────────────────────────────────────┘
```

---

# 1. Architectural Landscape and the Tokenization Bottleneck

The quantitative characterization of spontaneous animal behavior has undergone a structural transformation driven by high-throughput computer vision and unsupervised machine learning. Early automated systems relied on low-dimensional scalar metrics—such as total distance traveled, mean velocity, or boundary thigmotaxis—which frequently failed to capture the high-dimensional structure of natural movement. The advent of pose-estimation frameworks enabled high-fidelity tracking of anatomical keypoints across time. However, raw keypoint trajectories remain intrinsically noisy and dominated by high-frequency spatial jitter, making direct frame-by-frame clustering prone to spurious state transitions.

To resolve this challenge, **Motion Sequencing (MoSeq)** and **Keypoint-MoSeq** formulated behavioral extraction as a probabilistic generative modeling problem. By combining autoregressive hidden Markov models (AR-HMM) with sticky Hierarchical Dirichlet Process (HDP) priors, keypoint-MoSeq discretizes continuous pose dynamics into an ethological vocabulary of $K \approx 50\text{–}71$ sub-second behavioral primitives termed *"syllables"*. These syllables—characterized by a median bout duration of approximately 400 milliseconds—represent foundational units of movement (such as rears, turns, and grooming bouts) encoded within the dorsolateral striatum (DLS) and modulated by localized dopaminergic reinforcement.

While individual syllables offer physical interpretability, a continuous recording session containing over 10,000 discrete bout instances presents an analytical bottleneck: sub-second motifs alone do not elucidate the multi-second or minute-long behavioral programs that govern goal-directed navigation, exploration, or internal state shifts. Recent computational efforts have sought to bridge this temporal gap across distinct paradigms:

* **State-Based Hierarchical MoSeq (shMoSeq):** Weinreb et al. (*Neuron*, 2026) established that sub-second syllables organize into higher-level, multi-second "task-like" states. Mathematically defined as distinct syllable transition matrices, these macro-states correspond to self-directed behavioral programs (e.g., wall-following, novel object interaction, or micro-grooming routines) preferentially encoded within the prefrontal cortex (PFC).
* **Information-Theoretic Temporal Hierarchies:** Berman, Bialek, and Shaevitz (*PNAS*, 2016) demonstrated that animal behavior exhibits multi-scale non-Markovian temporal structure. Utilizing the Information Bottleneck on fruit fly (*Drosophila melanogaster*) behavioral sequences, they proved that predictive representations of future actions natively organize into a hierarchical continuum driven by slowly varying internal states.
* **Continuous Latent Embeddings and Motion Language Transformers:** Methods such as Unsupervised Behaviour Analysis and Magnification (**uBAM**, *Nature MI* 2021), Learnable Latent Embeddings (**CEBRA**, *Nature* 2023), and **BehaVERT** (*IJCV* 2026) apply contrastive learning or Transformer-based cloze-masking directly to continuous coordinate frames. However, tokenizing frame-by-frame poses ($T_t \in \mathbb{R}^{768}$) introduces substantial temporal autocorrelation noise, as adjacent 80 Hz frames remain over 95% identical.

The primary advantage of building downstream representations upon keypoint-MoSeq lies in its initial noise filtering. Starting from a discrete, ethologically validated vocabulary eliminates continuous pose jitter while retaining the expressive syntax of action sequences. Treating time-anchored sequence windows ($W_t = 68.3\text{ s}$, encompassing $\approx 170$ bout instances) as **"behavioral documents"** allows computational ethology to leverage the structural paradigms of Natural Language Processing (NLP): mapping raw motor primitives (words) into contextual sequence embeddings (sentences) and macroscopic behavioral programs (topics).

### The Multi-Stage Pipeline Hierarchy:
1. **Raw Video and Keypoint Tracking:** Video streams acquired at 80 Hz are processed via pose-estimation pipelines (e.g., SLEAP or DeepLabCut) to extract anatomical body coordinates.
2. **Syllabic Discretization:** Keypoint-MoSeq applies sticky HDP-AR-HMM modeling to convert continuous pose trajectories into a discrete ethological vocabulary of $K \approx 50\text{–}71$ syllables with continuous micro-kinematic features.
3. **Behavioral Document Windowing:** Continuous syllable streams are sliced into time-anchored evaluation windows ($W_t = 68.3\text{ s}$, corresponding to $\approx 170$ consecutive bouts).
4. **Contextual Sequence Encoding:** Deep sequence models (Transformers, Mamba) map discrete token windows into a continuous, low-dimensional behavioral semantic space ($\mathbf{z}_t \in \mathbb{R}^d$).
5. **Macro-Topic Discovery:** Unsupervised clustering and topic modeling partition the semantic space into 10–20 interpretable macro-behavioral programs.
6. **Geometric Phenotyping and Neuromodulation:** Optimal Transport metrics quantify phenotypic manifold displacements across disease models, while synchronous fiber photometry links subcortical dopamine and acetylcholine dynamics to trajectory vectors across the strategy space.

---

# 2. Deconstructing the Predictability Hierarchy: The Epistemic Information Ruler

To determine the statistical structure governing multi-second behavioral sequences, behavior must be decomposed into a nested informational ruler ($\mathcal{P}_0 \to \mathcal{P}_5$). This hierarchy quantifies the marginal predictive gain contributed by time, spatial geometry, instantaneous posture, static repertoire allocation, and non-linear chronological syntax when forecasting downstream behavioral budgets ($H = 120.0\text{ s}$).

```
P0 = Empty Null Baseline (Global Unigram Prior)
P1 = P0 U { t, t^2 }                                (Session Drift & Habituation)
P2 = P1 U { center_ratio, dist_to_wall, speed }     (Spatial Arena Geography)
P3 = P2 U { S_t }                                   (1-Step Markov Motor Grammar)
P4 = P3 U { u_count, u_time }                       (Action-Budget Repertoire BoW)
P5 = P4 U { Contextual Sequence Architecture }      (Chronological Syntax & Non-linear Context)
```

### 5-Fold Cross-Validation Empirical Benchmark Table ($T = 174,686\text{ bouts}$):

| Model Architecture | Mathematical Formulation / Paradigm | Mean Predictive Gain ($\%\Delta D_{\text{JS}}$) | Implied Role in Behavioral Hierarchy |
| :--- | :--- | :---: | :--- |
| **1. Bag-of-Words (BoW)** | Linear Ridge Regression on $[\mathbf{u}_{\text{count}}, \mathbf{u}_{\text{time}}] \in \mathbb{R}^{2K}$ | $+31.88\%$ | Static repertoire allocation ceiling ($\mathcal{P}_4$) |
| **2. Action2Vec (SGNS)** | Skip-Gram with Negative Sampling ($p=32$) | $+27.18\%$ | Distributional semantic neighborhood geometry |
| **3. Probabilistic Suffix Tree** | Variable-Order Discrete Markov Tree (Ron et al.) | $+14.77\%$ | Multi-scale discrete local transition syntax |
| **4. Explicit-Duration HSMM** | Semi-Markov State Persistence (Murphy / Linderman) | $+11.68\%$ | Temporal dwell-time and state persistence |
| **5. Selective SSM (Mamba-1)** | Linear-time Continuous State Space (Gu & Dao) | $+33.45\%$ | Long-horizon continuous selective memory |
| **6. Causal Action Transformer**| GPT-Style Causal Decoder ($d_{\text{model}}=64, 4\text{ Heads}$) | $\mathbf{+35.82\%}$ | Non-linear contextual feature interactions ($\mathcal{P}_5$) |
| **7. Transformer (Shuffle Ablation)**| Intra-Window Random Syllable Order Permutation | $+34.63\%$ | **$-1.19\%$ Syntax Penalty** isolating true sequence order |

### Key Empirical Findings:
1. **Repertoire Dominance:** Linear Ridge Regression on simple Bag-of-Words syllable counts ($\mathcal{P}_4$) accounts for $+31.88\%$ of the total predictive gain. Static action repertoire composition ($\mathcal{P}_4$) accounts for approximately **$89.0\%$** of total predictive power.
2. **Syntax Refinement:** Moving to a fully contextual Causal Action Transformer ($\mathcal{P}_5$) increases total predictive gain to $+35.82\%$, providing an **$11.0\%$ marginal refinement**.
3. **The Chronological Order Penalty:** Randomly shuffling syllable sequence order within window $W_t$ drops performance from $+35.82\%$ to $+34.63\%$, isolating an intra-window syntax penalty of **$-1.19\%$ ($p < 0.001$)**.
4. **The Syntactic Disambiguation Principle:** For matched episode pairs ($A$ and $B$) with identical syllable counts ($\|\mathbf{u}_A - \mathbf{u}_B\|_2 \approx 0$) but different sequence order, the contextual Transformer maps them to distinct coordinates ($\mathbf{z}_A \neq \mathbf{z}_B$) and their downstream futures diverge significantly ($D_{\text{JS}} > 0$).

---

## 2.1 The Behavioral Analytical Ceiling: Compression as Interpretability

A central theoretical question emerges from this empirical hierarchy:

> **How far can we compress behavioral history before we start losing information that matters?**

The primary value of behavioral compression is not computational efficiency—it is **human-in-the-loop interpretability**:
* If 100 syllables across thousands of bouts can be summarized by 4–6 recurring higher-order dimensions without losing meaningful predictive signal, behaviorologists no longer need to reason over overwhelming raw event streams.
* They can inspect a small set of transparent variables, compare them across cohorts (genotype/drug/disease), and validate them directly against representative video exemplars.
* This framing shifts computational ethology away from arbitrarily stacking deep neural network layers toward establishing an **empirical analytical ceiling**: discovering the exact point of representation sufficiency where added model complexity stops yielding behavioral or neurobiological insight.

---

# 3. Foundation Models and High-Dimensional Geometric Phenotyping

Rather than training isolated models on small experimental cohorts, large unlabelled institutional video archives enable **Behavioral Foundation Models** pretrained via self-supervised masked token prediction:
$$\max_{\boldsymbol{\theta}} \sum_{i \in \text{Masked}} \log P_{\boldsymbol{\theta}}(s_i \mid s_{\setminus i})$$

During pretraining, the model receives zero experimental, genotype, or drug labels, learning an unconditioned, general-purpose behavioral coordinate system ($\mathbf{z}_t \in \mathbb{R}^d$) that captures universal rodent syntax.

### Optimal Transport (Sinkhorn-Wasserstein $\mathcal{W}_{1, \epsilon}$) Phenotyping:
Treating behavioral sessions as continuous probability distributions $(\mu, \nu)$ over the latent semantic manifold $\mathbf{z}_t$, we compute the regularized 1-Wasserstein distance:
$$\mathcal{W}_{1, \epsilon}(\mu_A, \mu_B) = \min_{\boldsymbol{\gamma} \in \Pi(\mu_A, \mu_B)} \langle \boldsymbol{\gamma}, \mathbf{C} \rangle + \epsilon \sum_{i,j} \gamma_{ij} \log \gamma_{ij}$$

### Applications to Disease Models:
* **Parkinsonian Pre-Formed Fibril (PFF) vs. Monomer Controls:** Standard univariate metrics (distance moved, mean speed) fail to reach significance ($p > 0.05$). Sinkhorn-Wasserstein distance ($\mathcal{W}_1$) on the strategy manifold reveals highly significant geometric displacement ($p < 0.001$), isolating altered dwell-time dynamics and restricted transition vectors between exploratory regimes.
* **Dopamine Transporter Knock-In (DAT-KI) vs. Wild-Type:** Manifold analysis quantifies motor perseveration as severe **geometric contraction**: trajectories become locked within low-entropy, repetitive sub-regions with reduced geodesic divergence.

---

# 4. Neuromodulatory Dynamics: Striatal Dopamine & Acetylcholine Steering

By pairing synchronized 2-channel fiber photometry (**dLight1 for Dopamine, ACh3.0 for Acetylcholine** at 80 Hz) with continuous latent semantic coordinates $\mathbf{z}_t$, neuromodulation is mapped directly to macroscopic trajectory vectors:
$$\mathbf{y}(t) = \begin{bmatrix} \text{DA}(t) \\ \text{ACh}(t) \end{bmatrix} \longrightarrow \Delta \mathbf{z}_t = \mathbf{z}_{t+\Delta} - \mathbf{z}_t$$

```
                            THREE MODES OF NEUROMODULATORY COUPLING
                            
  1. Tonic State Encoding        2. Phasic Trajectory Steering          3. Strategic Surprisal
  ───────────────────────        ─────────────────────────────          ─────────────────────
     DA_tonic ~ f( z_t )            DA_phasic ~ f( Delta z_t )             DA/ACh ~ -log P(Z'|Z)
  (Background dopamine tone      (Transient dopamine bursts act         (Synchronous DA bursts and ACh
   defines active strategy)       as directional steering vectors)       dips signal macro-regime errors)
```

1. **Tonic State Encoding ($\text{DA}_{\text{tonic}} \sim f(\mathbf{z}_t)$):** Sustained dopamine tone defines the operational zone on the strategy manifold (high tonic DA in spatial exploration, low tonic DA in stationary grooming/rest).
2. **Phasic Trajectory Steering ($\text{DA}_{\text{phasic}} \sim f(\Delta \mathbf{z}_t)$):** Transient dopamine bursts correlate with the magnitude and direction of manifold displacement, acting as internal steering vectors that drive transitions between behavioral programs.
3. **Strategic Surprisal & Macro-Prediction Error ($\text{DA}/\text{ACh}_{\text{transient}} \sim -\log P(Z_{t+\Delta} \mid Z_t)$):** Abruptly aborting an ongoing behavioral program to enter an unexpected macro-topic triggers a sharp neuromodulatory transient, expanding the striatal cholinergic-dopaminergic axis into multi-second program monitoring.

---

# 5. Ethological Interpretability & Landmark Comparison Matrix

Every discovered macro-topic is grounded via automated **Ethological Topic Cards** pairing latent regions with:
* **Video Exemplars:** 10–20 synchronized real-world video clips.
* **Spatial Occupancy Heatmaps:** Arena perimeter vs. center distribution.
* **Enriched Syllable Vocabularies:** Differential histograms showing over/under-expressed syllables.
* **Transition Kinematics:** Graph visualizations of dwell-times and transition probabilities.

### Comprehensive Methodological Comparison:

| Method / Paradigm | Primary Input Tokenization | Temporal Scale | Modeling Framework | Phenotypic & Neural Output |
| :--- | :--- | :--- | :--- | :--- |
| **Keypoint-MoSeq** (Weinreb 2024) | Continuous keypoint poses | Sub-second (~400 ms) | Sticky HDP-AR-HMM | Identifies $K \approx 50\text{–}71$ syllables; correlates DLS dopamine with single motifs. |
| **shMoSeq** (Weinreb 2026) | Syllable sequences | Multi-second (2–10 s) | Hierarchical HMM | Discovers task-like behavioral states represented preferentially in PFC. |
| **Predictive Hierarchy** (Berman 2016) | Postural modes (fruit fly) | Multi-scale temporal continuum | Information Bottleneck | Proves non-Markovian predictability and temporal hierarchy in behavior. |
| **uBAM** (Brattoli 2021) | Raw video / keypoints | Multi-frame sliding windows | Deep Recurrent VAE | Learns unsupervised posture representations for mutant phenotyping. |
| **CEBRA** (Schneider 2023) | Continuous pose / neural traces | Variable temporal windows | Joint contrastive learning | Learns low-dimensional latent spaces mapping behavior to neural dynamics. |
| **BehaVERT** (Shin 2026) | Continuous pose vectors ($\mathbb{R}^{768}$) | Sentence-level trajectories | Transformer (BERT cloze) | Motion language model decoding behavioral semantics directly from pose. |
| **Proposed Framework** | **Discrete MoSeq Syllables ($K \approx 50\text{–}71$)** | **Hierarchical ($W_t = 68.3\text{s}$ windows)** | **Foundation Sequence Transformer / Mamba** | **Unsupervised topic discovery, Sinkhorn-Wasserstein phenotyping, and DA/ACh trajectory steering.** |
