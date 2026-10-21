# Behavioral Sentiment & Latent Strategy Clustering (Version 2)

This document outlines the conceptual, mathematical, and implementation framework for **Phase 1.b: Unsupervised Behavioral Strategy Clustering**. 

---

## 1. Introduction: From Rigid Sentences to Continuous "Vibes"

In our previous attempt (Phase 1.a), we attempted to segment syllable sequences into discrete "sentences" using slow-syllable punctuation boundaries. While mathematically elegant, treating mouse behavior like a human text document introduces rigid grammatical assumptions:
1. It assumes there are clear, discrete boundaries (e.g., periods/commas) that separate behavioral thoughts.
2. It discards the continuous tempo, physical dynamics, and smooth transitions that define animal movement.

To overcome this, **Version 2** moves away from sentence segmentation. Instead, we model behavior as a **continuous discourse** and analyze it using a **sliding window**. At any given moment, the animal is executing a latent **"behavioral strategy"** or **"vibe"**—a state that is larger than an individual syllable or trigram, but smaller than the entire session. By representing this window using continuous physical and semantic descriptors, we can cluster these windows into discrete behavioral strategies (e.g., "active exploration," "anxious thigmotaxis," or "stereotyped turning").

---

## 2. The Speech Analogy of Behavior

To extract the "sentiment" or "vibe" of a speech at a given moment, NLP models look at three parallel channels: **Prosody**, **Semantics**, and **Syntax**. We can map this directly to keypoint-derived behavioral streams:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                    SLIDING WINDOW                      │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
               ┌─────────────────────────────┼─────────────────────────────┐
               ▼                             ▼                             ▼
       [1. MOVEMENT PROSODY]         [2. SYLLABLE SEMANTICS]       [3. TRANSITION SYNTAX]
       "How it is being said"         "What is being said"         "Grammatical complexity"
               │                             │                             │
    • Velocity (mean/std)         • Behavior2Vec Embedding      • POS Histogram (G=5)
    • Angular Velocity (mean/std)   (PMI-SVD 16D space)         • Transition Rate
    • Acceleration (mean/std)                                   • Sequence Entropy
    • Distance from Arena Center
    • Syllable Duration Std
               └─────────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Hybrid Feature Vector    │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  UMAP Dimension Reduction │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     Clustering Space      │
                               │    (HDBSCAN / GMM)        │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   Latent Strategy IDs     │
                               └───────────────────────────┘
```

### A. Movement Prosody (How it is said)
* **Speech Analogy**: Pitch, volume, pacing, and pause duration convey emotion independently of the words chosen (e.g., shouting quickly conveys high arousal/excitement).
* **Behavioral Mapping**: Continuous frame-level kinematics and physical tempo.
* **Features**:
  - **Kinematics**: Mean and standard deviation of `velocity_px_s`, `angular_velocity`, and `acceleration` within the window.
  - **Arena Context**: Mean `dist_from_center` (capturing center exploration vs. thigmotaxis).
  - **Temporal Pacing**: Standard deviation of syllable durations (capturing rhythm/regularity vs. arrhythmia).
  - *Note*: Mean syllable duration is omitted here because it is the reciprocal of transition rate, preventing double-weighting of behavioral "chunkiness."

### B. Syllable Semantics (What is said)
* **Speech Analogy**: Word selection determines the theme. We can represent the "vibe" of a speech snippet by averaging the word embeddings (capturing their semantic meaning and associations).
* **Behavioral Mapping**: The type of syllables occurring and their contextual meaning.
* **Features**:
  - **SVD Embeddings**: We train `Behavior2Vec` using Singular Value Decomposition (SVD) on the Pointwise Mutual Information (PMI) co-occurrence matrix of syllable transitions. The embedding dimension $D$ (default 16) is selected via a singular-value scree plot.
  - **Window Average**: The semantic vector $\vec{v}_{\text{semantics}}$ of the window is the mean embedding of all syllables present.
  - *Note*: Sparse global syllable velocities are dropped to avoid redundancy and high dimensionality.

### C. Transition Syntax (The style and complexity)
* **Speech Analogy**: Short, simple sentences denote urgency/fear, whereas complex clauses denote calm descriptions.
* **Behavioral Mapping**: Grammatical complexity, parts of speech, and transition pacing.
* **Features**:
  - **Part-of-Speech (POS) Histograms**: We cluster the $D$-dimensional syllable embeddings using K-Means into $G$ functional groups (e.g., Locomotion, Grooming, Rearing, Inactive, Turning). The value of $G$ is chosen via silhouette score sweeps. The window's syntax is represented by the histogram distribution of these $G$ categories.
  - **Transition Rate**: The number of syllable transitions per second (capturing structural pacing).
  - **Sequence Entropy**: Shannon entropy of the syllable transition matrix within the window (quantifying behavioral predictability vs. chaos).

---

## 3. Twofold Window Length Optimization: Discriminability vs. Coherence

To find the optimal window size $W$ (in seconds), we evaluate a mathematical trade-off between two opposing normalized metrics over a sweep of $W \in [2\text{s}, 60\text{s}]$, controlling for geometric overlap artifacts:

### Metric 1: Discriminability Index ($D(W)$) — *Monotonically Decreasing*
* **Definition**: The total variance of the window features across the session (measured as the trace of the covariance matrix):
  $$D(W) = \text{trace}\left(\text{Cov}(\vec{X}_{\text{window}})\right)$$
* **Rationale**: Small windows are sparse and highly variable. As $W \to \text{session length}$, every window averages over the whole session, converging to the global mean, causing inter-window variance to drop to 0. We want to maximize this to keep strategies distinct.

### Metric 2: Temporal Coherence Index ($C(W)$) — *Monotonically Increasing*
* **Definition**: The self-transition probability of strategy states over time:
  $$C(W) = P(S_{t+1} = S_t)$$
  where $S_t$ is the Strategy ID assigned to the window at step $t$.
* **Anti-Artifact Control**: To prevent $C(W)$ from being artificially inflated by window overlap geometry, we evaluate $C(W)$ **exclusively on adjacent, non-overlapping windows** during the sweep (e.g., comparing labels of disjoint blocks: 0–10s vs 10–20s). 
* **Null Control**: We run the same sweep on a time-shuffled window control to calculate a baseline coherence $C_{\text{null}}(W)$ and compute a chance-corrected coherence:
  $$\text{Coherence}_{\text{corrected}}(W) = \frac{C(W) - C_{\text{null}}(W)}{1 - C_{\text{null}}(W)}$$

### The Sweet Spot (Optimization Point)
We normalize $D(W)$ and $C_{\text{corrected}}(W)$ to $[0, 1]$ over the sweep range. The **optimal window size** $W_{\text{opt}}$ is the intersection point where:
$$D(W_{\text{opt}}) = C_{\text{corrected}}(W_{\text{opt}})$$
Alternatively, we select $W_{\text{opt}}$ to maximize the harmonic mean (F-score) of the two metrics.

---

## 4. Time-Based vs. Step-Based Windows

We prioritize **Time-Based Windows** (T seconds) over Step-Based Windows (N transitions) for the following reasons:
1. **Kinematic Integrity**: Kinematics (velocities, acceleration) are physical rate-based metrics (e.g., px/s). A time-based window keeps their physical definition intact.
2. **Photometry Alignment**: Dopamine signals (dLight) occur in physical real-time. Distorting physical time with step-based windows makes alignment with calcium dynamics impossible.
3. **Handling Idleness**: When the mouse freezes or rests, step-based windows span a huge time scale, blending distinct phases of behavior. Time-based windows naturally isolate quiet states.
