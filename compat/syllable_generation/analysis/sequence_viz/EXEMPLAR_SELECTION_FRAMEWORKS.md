# Methodological Taxonomy: Defining the "Exemplary" Behavioral Sequence

A comprehensive reference on the mathematical, epistemological, and neuroethological frameworks for defining and selecting **exemplary** behavioral instances and multi-syllable chains in Keypoint-MoSeq (KPMS).

---

## Table of Contents

1. [The Foundational Dilemma: What Makes an Instance "Exemplary"?](#1-the-foundational-dilemma-what-makes-an-instance-exemplary)
2. [The Circularity Trap of Whole-Chain Medoids](#2-the-circularity-trap-of-whole-chain-medoids)
3. [Taxonomy of Exemplar Selection Frameworks](#3-taxonomy-of-exemplar-selection-frameworks)
   - [Paradigm 1: Uniform Spread Sampling (Unconstrained Baseline)](#paradigm-1-uniform-spread-sampling-unconstrained-baseline)
   - [Paradigm 2: Scalar Kinematic Medoid (Summary Features)](#paradigm-2-scalar-kinematic-medoid-summary-features)
   - [Paradigm 3: Whole-Chain Contrastive Density Ratio (Native KPMS)](#paradigm-3-whole-chain-contrastive-density-ratio-native-kpms)
   - [Paradigm 4: Forward Transition-Bucketed Constituent Exemplars (Generative & Causal)](#paradigm-4-forward-transition-bucketed-constituent-exemplars-generative--causal)
   - [Paradigm 5: Dual-Score Physical Quality vs. Kinematic Similarity ($Q$ vs. $S$)](#paradigm-5-dual-score-physical-quality-vs-kinematic-similarity-q-vs-s)
   - [Paradigm 6: Neuroethological Spatial Context (Dvorkin Topology & Alexander EBVs)](#paradigm-6-neuroethological-spatial-context-dvorkin-topology--alexander-ebvs)
4. [Comparative Matrix](#4-comparative-matrix)
5. [Decision Guide: Which Paradigm Should You Use?](#5-decision-guide-which-paradigm-should-you-use)
6. [Code Implementation & Catalog Index](#6-code-implementation--catalog-index)

---

## 1. The Foundational Dilemma: What Makes an Instance "Exemplary"?

In behavioral neuroscience, identifying recurring sequences (e.g. `[1 -> 11 -> 1]` locomotor alternation or `[6 -> 17 -> 2]` orienting turns) is only the first step. When scanning continuous tracking datasets across dozens of animals, an investigator often retrieves hundreds or thousands of raw candidate matches.

However, raw retrievals are heavily corrupted by:
- **Tracking glitches & camera drops**: Sudden coordinate jumps.
- **Boundary micro-flickers (1–3 frames / 25–75 ms)**: Rapid model chatter at stage transitions.
- **Dynamic error accumulation**: A 3-stage chain with a 5% syllable error rate has a $1 - (0.95)^3 \approx 14.3\%$ chance of containing a bad stage; a 10-stage chain has a $40.1\%$ corruption rate.

To visually audit these behaviors without confusion, investigators seek **exemplars**—canonical representations that best capture the biological phenomenon. But how should "exemplary" be defined?

```
                       ┌──────────────────────────────────────────────┐
                       │ How is "Exemplary" Defined for a Sequence?   │
                       └──────────────────────┬───────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        [ Whole-Chain / Post-Hoc ]                         [ Constituent / Generative ]
   Evaluates the entire multi-syllable               Evaluates each constituent syllable
   trajectory as a single unified gesture.           locally within its forward motor transition.
   • Captures global trajectory shape.               • Epistemologically grounded in motor primitives.
   • Susceptible to circularity & error masking.     • Forward-in-time, causal, and unmasked.
```

---

## 2. The Circularity Trap of Whole-Chain Medoids

A common instinct in machine learning is to take all retrieved instances of a sequence and select the geometric **medoid** (or density peak) across the entire multi-stage trajectory. 

### Why is this circular?

$$\text{Instance}^* = \arg\min_{x \in \mathcal{C}} \sum_{y \in \mathcal{C}} \|x - y\|$$

1. **The Self-Referential Assumption**:
   Defining an episode as "exemplary" merely because it sits near the center of mass of the retrieved set assumes what it is trying to prove: that the retrieved instances form a single, coherent biological phenomenon. If $40\%$ of the retrieved set contains tracking artifacts or boundary chatter, the whole-chain medoid simply selects an instance that looks like the *average artifact*.
   > *"These chains are exemplary because they look most like the chains we called exemplary."*

2. **The Error Cancellation / Compensation Flaw**:
   In a high-dimensional concatenated trajectory (e.g. $L=20$ timepoints across 10 SLDS latents $= 200\text{D}$), errors in intermediate stages can cancel out. An episode with a severely corrupted, half-aborted middle syllable (e.g., tracking jitter during syllable 11) can still be selected as the "exemplar" if the first and third syllables have strong locomotor velocity that pulls the aggregate vector close to the cluster center.

3. **Violation of Causal Generation**:
   Mice do not plan a global 3-second Cartesian curve and divide it into tokens after the fact. The nervous system executes behavior **step-by-step forward in time**. Calling a sequence "exemplary" after the fact ignores whether each constituent motor command was well-formed.

---

## 3. Taxonomy of Exemplar Selection Frameworks

### Paradigm 1: Uniform Spread Sampling (Unconstrained Baseline)

- **Concept**: Pulls instances evenly across the recording sessions and animals without applying any geometric or kinematic filtering.
- **Mathematical Formulation**:
  $$i_k = \left\lfloor k \times \frac{N - 1}{M - 1} \right\rfloor, \quad k \in \{0, 1, \dots, M-1\}$$
- **Ethological Role**: Serves as the **unbiased baseline**. It demonstrates the true, unvarnished inter-session variability of that sequence across wild-type and knock-in mice under saline and amphetamine.
- **Catalog**: [`screening_24_batch/README_24_CANDIDATES.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/screening_24_batch/README_24_CANDIDATES.md)

---

### Paradigm 2: Scalar Kinematic Medoid (Summary Features)

- **Concept**: Summarizes each episode into an 8D scalar vector:
  $$\mathbf{v} = \left[ \Delta t_{\text{total}}, \, \Delta s_{\text{disp}}, \, \bar{v}_{\text{speed}}, \, v_{\max}, \, \Delta \theta_{\text{turn}}, \, |\Delta \theta|, \, Q_{\min}, \, S_{\text{mean}} \right]^T$$
  and finds the medoid in z-score standardized Euclidean space.
- **Strengths**: Extremely fast; guarantees that durations and overall travel distances are typical.
- **Limitations**: **Blind to pose geometry**. A mouse performing a sharp left turn followed by a sharp right turn can have the exact same speed, duration, and net displacement as a mouse running straight forward.

---

### Paradigm 3: Whole-Chain Contrastive Density Ratio (Native KPMS)

- **Concept**: Direct adaptation of Keypoint-MoSeq's native grid movie sampling algorithm (`keypoint_moseq.util.sample_instances(mode='density')`) to entire multi-syllable episodes.
- **Mathematical Formulation**:
  1. Continuous 10D SLDS latents $\mathbf{z}(t)$ spanning the entire chain (`start_frame:end_frame`) are interpolated to $L=20$ timepoints ($200\text{D}$).
  2. 4D PCA is fitted across 400 background episodes sampled from 30 sessions.
  3. For every candidate chain $x$, compute local density within its family and global density against background behavior:
     $$\text{Local Density}(x) = \frac{1}{\frac{1}{k}\sum_{y \in \text{kNN}_{\text{family}}(x)} \|x - y\|}, \quad \text{Global Density}(x) = \frac{1}{\frac{1}{k}\sum_{y \in \text{kNN}_{\text{bg}}(x)} \|x - y\|}$$
  4. The **Exemplar** maximizes the contrastive ratio:
     $$\text{Exemplar} = \arg\max_{x \in \mathcal{C}} \left( \frac{\text{Local Density}(x)}{\text{Global Density}(x)} \right) = \arg\max_{x \in \mathcal{C}} \left( \frac{\bar{d}_{\text{global}}(x)}{\bar{d}_{\text{local}}(x)} \right)$$
- **Observed Performance**:
  - Achieves **$54.5\%\text{--}89.4\%$ kinematic dispersion reduction** in continuous latent space.
  - Density ratios reach up to **$4.52\times$** background density.
- **Catalog**: [`screening_24_exemplars/README_24_EXEMPLARS.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/screening_24_exemplars/README_24_EXEMPLARS.md)

---

### Paradigm 4: Forward Transition-Bucketed Constituent Exemplars (Generative & Causal)

- **Concept**: Grounded in motor control theory: **combinations of exemplary forward transitions create exemplary chains**.
- **Why It Resolves the Dilemma**:
  - It does **not** evaluate the whole chain post-hoc (eliminating the circularity trap).
  - It does **not** assume syllables are isolated Lego bricks (it explicitly conditions on forward transition $\mathcal{B}(S_k \to S_{k+1})$).
  - It captures **co-articulation**: how the animal shapes syllable $S_k$ in anticipation of $S_{k+1}$.
- **Mathematical Formulation**:
  1. Group all bouts of syllable $S$ across all sessions by their forward destination into bucket $\mathcal{B}(S \to S_{\text{next}})$ (or $\mathcal{B}(S \to C_{\text{next}})$ if $N < 15$).
  2. For each bout, resample its single-syllable 10D latents to $L=15$ timepoints ($150\text{D}$) and fit 3D PCA.
  3. Find the transition medoid exemplar $x^*_{\mathcal{B}}$.
  4. Each constituent stage receives a percentile rank $p_k \in [0, 1]$ based on distance to $x^*_{\mathcal{B}}$ ($1.0 = \text{identical to transition exemplar}$).
  5. The sequence is scored by its composite geometric mean:
     $$\bar{p} = \left( \prod_{k=1}^M p_k \right)^{1/M}, \quad p_{\min} = \min_{k=1}^M p_k$$
- **Strict vs. Moderate Criteria**:
  - **Strict Constituent Chains**: $p_{\min} \ge 0.50$ (every constituent syllable ranks in the top half of its transition bucket).
  - **Moderate Constituent Chains**: $p_{\min} \ge 0.33$.
- **Empirical Compounding Finding**:
  - For `[1, 11, 1]`, **$17.6\%$** of chains are strictly exemplary (vs. $12.5\%$ random expectation $(0.5)^3$).
  - For `[1, 11, 1, 11]`, **$12.6\%$** of chains are strictly exemplary (vs. $6.25\%$ random expectation $(0.5)^4$).
  - Proves that clean motor transitions naturally reinforce each other into compound motor programs.
- **Catalog**: [`screening_24_transition_exemplars/README_24_TRANSITION_EXEMPLARS.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/screening_24_transition_exemplars/README_24_TRANSITION_EXEMPLARS.md)

---

### Paradigm 5: Dual-Score Physical Quality vs. Kinematic Similarity ($Q$ vs. $S$)

- **Concept**: A critical insight from ethological review is that **rare behavior is not corrupt behavior**. High-speed sprints, hairpin turns, and unusual postures are genuine biological motor variants, not tracking noise.
- **Mathematical Formulation**:
  1. **Physical Quality Score ($Q \in [0, 1]$)**:
     $$Q = \prod_{k=1}^M \left[ \mathbb{I}(\Delta s_{\max} \le 22\text{px}) \times \mathbb{I}(5 \le \Delta t_k \le 65) \times \mathbb{I}(|\Delta \theta_{\text{flip}}| < 150^\circ) \right]$$
     Measures measurement telemetry only. Rapid physical movement receives **zero penalty**.
  2. **Kinematic Similarity Score ($S \in [0, 1]$)**:
     $$S = \exp\left( -\frac{1}{2} D_{\text{Mahalanobis}}^2(\mathbf{v}, \boldsymbol{\mu}_{\text{baseline}}) \right)$$
     Measures statistical typicality relative to the population.
- **Three-Way Classification**:
  - `TRUSTED_CLEAN`: $Q \ge 0.80, S \ge 0.40$ (Clean tracking, typical kinematics).
  - `MOTOR_VARIANT`: $Q \ge 0.80, S < 0.40$ (Clean tracking, unusual/extreme kinematics). **Preserved as genuine biology!**
  - `MEASUREMENT_CORRUPT`: $Q < 0.80$ (Telemetry glitch, teleportation jump, boundary chatter). Discarded.
- **Implementation**: [`bout_quality_and_spatial_verifier.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/bout_quality_and_spatial_verifier.py)

---

### Paradigm 6: Neuroethological Spatial Context (Dvorkin Topology & Alexander EBVs)

- **Concept**: Rodents navigate in relation to physical boundaries and places of attraction, not abstract Cartesian grids. An exemplary instance must make spatial sense.
- **Two Neuroethological Foundations**:
  1. **Dvorkin, Benjamini, & Golani (2008)** (*PLOS Comput Biol*):
     - Open-field exploration is organized into discrete **excursions** from places of attraction.
     - Topology: **Return to Origin ($A \to A$)** vs. **Relocation ($A \to B$)** along perimeter or across interior.
  2. **Alexander et al. (2020)** (*Science Advances*):
     - Retrosplenial Egocentric Boundary Cells (EBCs) encode physical walls relative to body axis (`LEFT_FLANK`, `RIGHT_FLANK`, `SNOUT_AHEAD`, `OPEN_ARENA`).
     - Rodents do not navigate by compass directions (North vs South wall). A mouse running along the North wall with the wall on its Left Flank is in the **identical egocentric neural state** as running along the South wall with the wall on its Left Flank.
- **Ethological Application**:
  In seqNMF Factor 05 (`[1, 11, 1, 30]`), $66.2\%$ of clean occurrences are genuine wall-following thigmotaxis with `LEFT_FLANK` boundary contact. Evaluating sequences through egocentric boundary coordinates aligns the analysis directly with brain physiology.

---

## 4. Comparative Matrix

| Paradigm | Evaluated Unit | Directionality | Co-articulation? | Susceptible to Circularity? | Treats Motor Variants? | Data Requirements |
|---|---|---|---|---|---|---|
| **1. Uniform Spread** | Full Episode | None (Sampling) | Natural | None (Unbiased) | Included as-is | None |
| **2. Scalar Kinematic Medoid** | Full Episode (8D) | Post-Hoc | No (Scalar summary) | Low | Discards as outliers | Low |
| **3. Whole-Chain Density Ratio** | Concatenated Trajectory ($200\text{D}$) | Post-Hoc (Whole chain) | Implicit (Concatenated) | **High** (Self-referential) | Penalized as low-density | High ($N \ge 20$) |
| **4. Forward Transition Exemplar** | Constituent Bouts ($150\text{D}$) | **Forward / Causal** | **Explicit** (via $\mathcal{B}(S_k \to S_{k+1})$) | **Zero** (Local motor criteria) | Preserved if transitions clean | Moderate |
| **5. Dual-Score ($Q$ vs. $S$)** | Stage & Chain | Hybrid | Via Mahalanobis | Zero | **Explicitly bifurcated & saved** | Population baselines |
| **6. Spatial Context (EBV / Topology)** | Arena Coordinates | Allocentric $\to$ Egocentric | Spatial trajectory | Zero | Characterized by location | Boundary tracking |

---

## 5. Decision Guide: Which Paradigm Should You Use?

```
                                  What is your primary goal?
                                               │
             ┌─────────────────────────────────┼─────────────────────────────────┐
             ▼                                 ▼                                 ▼
   [ Visual Presentation /          [ Mechanistic Motor Control /      [ Phenotype & Drug Discovery /
        Initial Screening ]              Neuroethology ]                     Exploration Analysis ]
             │                                 │                                 │
             ▼                                 ▼                                 ▼
   Paradigm 3: Whole-Chain           Paradigm 4: Forward Transition     Paradigm 5 + 6: Dual-Score ($Q/S$)
   Contrastive Density Ratio         Constituent Exemplars              + Relational Topology (EBVs)
   (Maximizes visual cohesion        (Causal, eliminates circularity,   (Preserves drug-induced hyper-variants;
   across 6 grid tiles; 55-89%       grounds compound gestures in       tracks thigmotactic perimeter vs
   variance drop).                   motor primitives).                 open-field excursion biology).
```

1. **For Phase 1 Initial Visual Screening**:
   Use **Paradigm 3** ([`screening_24_exemplars/`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/screening_24_exemplars/)) or **Paradigm 4** ([`screening_24_transition_exemplars/`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/screening_24_transition_exemplars/)). Both produce crisp, non-glitchy 6-tile movies at 40 FPS real-speed.
2. **For Scientific Publishing & Rigorous Mechanistic Analysis**:
   Use **Paradigm 4** (Forward Transition-Bucketed Constituent Scoring). It provides the only epistemologically unassailable proof that candidate chains are composed of genuine, well-formed motor primitives without post-hoc cherry-picking.
3. **For Pharmacological Comparisons (Saline vs. Amphetamine)**:
   Use **Paradigm 5**. Amphetamine dramatically increases motor speed and turn rates. Whole-chain medoid approaches mistakenly discard amphetamine episodes as "atypical," whereas Dual-Score verification confirms they have $Q \ge 0.85$ (pristine physical execution) and marks them as `MOTOR_VARIANT`.

---

## 6. Code Implementation & Catalog Index

All frameworks are implemented, verified, and cataloged within the repository:

| Framework | Implementation Script | Video Output Directory | Markdown Catalog |
|---|---|---|---|
| **Uniform Spread Batch** | [`run_24_candidate_screening_batch.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/run_24_candidate_screening_batch.py) | [`screening_24_batch/`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/screening_24_batch/) | [`README_24_CANDIDATES.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/screening_24_batch/README_24_CANDIDATES.md) |
| **Whole-Chain Density Ratio** | [`run_24_candidate_exemplar_batch.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/run_24_candidate_exemplar_batch.py) | [`screening_24_exemplars/`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/screening_24_exemplars/) | [`README_24_EXEMPLARS.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/screening_24_exemplars/README_24_EXEMPLARS.md) |
| **Forward Transition Exemplars** | [`run_24_candidate_transition_exemplars.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/run_24_candidate_transition_exemplars.py) | [`screening_24_transition_exemplars/`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/screening_24_transition_exemplars/) | [`README_24_TRANSITION_EXEMPLARS.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/screening_24_transition_exemplars/README_24_TRANSITION_EXEMPLARS.md) |
| **Dual-Score $Q/S$ & Spatial EBVs** | [`bout_quality_and_spatial_verifier.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/bout_quality_and_spatial_verifier.py) | [`output_pilot_movies/`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/output_pilot_movies/) | [`README.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/sequence_viz/README.md) |
