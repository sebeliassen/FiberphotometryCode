# Strategy Evaluation Framework & Non-Circular Metrics Update

**Date:** Monday, August 17, 2026  
**Context:** Post-bug evaluation audit on clean active-mouse tracking data (14,870 windows across 30 sessions).

---

## 1. Executive Summary & Problem Formulation

Following the resolution of the **empty arena pre-entry bug** (where pre-session background noise created an artificial cluster, "Vibe 1", inflating 4-way classification accuracy to ~72%), decodability metrics on clean data settled into their true baselines:
* **Genotype Decodability (`WT` vs `DAT-KI`)**: **83.3% – 90.0%** (strong grammatical separation).
* **4-Way Cohort Decodability**: **~40.0% – 53.3%** (well above 25% chance).
* **Full-Session Drug Decodability**: **~36.7% – 55.4%** (diluted by pre-injection baseline).
* **Post-Injection Drug Decodability ($t = 20\text{--}60\text{ min}$)**: **~65.0% – 70.0%**.

While the model remains biologically informative, relying purely on supervised Random Forest accuracy across small session counts ($N=30$) creates an optimization trap: it pushes the model toward oversimplified $K=2\text{--}3$ trivial states (e.g., "active" vs "inactive") and ignores behavioral richness.

Furthermore, defining what makes an unsupervised strategy "good" is prone to **circular reasoning** (e.g., validating wall-hugging strategies by checking if the mouse is near the wall). This document records our evaluation of candidate validation metrics and formalizes our path forward.

---

## 2. Critique of Evaluation Dimensions

We evaluated four primary avenues for assessing strategy model quality and hyperparameter effectiveness:

```
                          ┌────────────────────────────────────────────────────────┐
                          │            CANDIDATE EVALUATION DIMENSIONS             │
                          └──────────────────────────┬─────────────────────────────┘
                                                     │
               ┌─────────────────────────┬───────────┴───────────┬─────────────────────────┐
               ▼                         ▼                       ▼                         ▼
        [ A. NEURAL (DA) ]        [ B. KINEMATICS (η²) ]  [ C. SYNTACTIC MOTIFS ]   [ D. PHARMA SHIFTS ]
        Photometry alignment      Effect size on speed/   Relaxed N-grams &         Pre-to-post injection
        around vibe switches      centroid distance       sequence perplexity       state transitions
               │                         │                       │                         │
        ⚠️ Confounded by           ⚠️ Partially circular;  ✅ Non-circular;          ✅ Non-circular;
        DAT mutation kinetics     speed follows syntax    information-theoretic     biological ground truth
```

### A. Neural / Dopamine Alignment (Fiber Photometry PETH)
* **Hypothesis**: True strategy boundaries should correlate with striatal dopamine ($DA(t)$ / dLight) transients or tonic shifts.
* **Critical Limitations**:
  1. **Non-Universal Gating**: Dopamine encodes specific variables (movement vigor, reward prediction error, novelty, incentive salience), not the boundary between every arbitrary behavioral state (e.g., shifting from wall-sniffing to grooming may elicit no DA response).
  2. **DAT-KI Mutation Confound**: DAT-KI mice possess a dopamine transporter knockdown mutation, causing chronically slowed clearance and altered baseline DA dynamics. Using DA alignment as an optimization loss risks overfitting the behavioral taxonomy to WT kinetics and penalizing valid mutant behaviors.
* **Verdict**: **Downstream Discovery Only**. Do not use DA alignment as an objective function for hyperparameter selection. Fit behavior independently first, then investigate how DA modulates those strategies across genotypes.

### B. Kinematic Encoding Effect Size ($\eta^2_{\text{speed}}$, $\eta^2_{\text{dist}}$)
* **Hypothesis**: A good strategy model should explain a large proportion of variance in continuous physical metrics (speed, turning, arena location).
* **Critical Limitations**:
  * **Partial Circularity**: Syllables are defined by pose trajectories. Transitions such as `run_left -> run_straight` mechanically imply high velocity. Finding high $\eta^2_{\text{speed}}$ simply confirms that clusters group locomotion-related syllables together.
* **Verdict**: **Sanity Check Baseline**. Useful for verifying that clusters are not white noise, but insufficient as an optimization target.

### C. Relaxed Syntactic Motifs & Sequence Perplexity (Intrinsic Metric)
* **Hypothesis**: A strategy represents a latent *behavioral topic* or *context-free grammar*. Within a true strategy, syllable sequences should exhibit structured, predictable syntax rather than random transitions.
* **Linguistic Translation**:
  * Avoid rigid $N$-gram matching ($V^N$ sparsity problem).
  * Use **relaxed/fuzzy phrase matching** and continuous embeddings (`Behavior2Vec` / transition manifolds) to identify semantically conserved behavioral templates.
  * Measure **Transition Perplexity**: A valid strategy significantly reduces conditional entropy $H(S_{t+1} \mid S_t, \text{Vibe}_k)$ relative to a time-shuffled null model.
* **Verdict**: **Strongest Intrinsic Mathematical Metric**. Completely non-circular because it evaluates the internal information-theoretic consistency of the sequence.

### D. Within-Animal Pharmacological Phase Transitions (Extrinsic Metric)
* **Hypothesis**: The strategy distribution vector $\vec{P} = [p(\text{Vibe}_1), \dots, p(\text{Vibe}_K)]$ should capture clean, low-dimensional phase shifts under pharmacological intervention.
* **Testing Protocol**:
  * **Saline Controls**: Pre-injection ($0\text{--}20\text{ min}$) to Post-injection ($20\text{--}60\text{ min}$) divergence $D_{\text{JS}}(P_{\text{pre}} \parallel P_{\text{post}})$ should remain low (behavioral stationarity / normal habituation).
  * **Amphetamine Treatment**: The drug triggers intense, stereotyped behavioral re-allocation; $D_{\text{JS}}(P_{\text{pre}} \parallel P_{\text{post}})$ should show a large, structured shift concentrated in specific hyper-aroused or locomotor vibes.
* **Verdict**: **Strongest Extrinsic Biological Metric**. Uses an experimental within-subject intervention rather than arbitrary post-hoc labels.

---

## 3. The Balanced Evaluation Matrix

Going forward, hyperparameter selection and model validation will use a **dual-anchor framework**:

| Metric Layer | Concrete Measurement | Role in Pipeline | Non-Circularity Rationale |
| :--- | :--- | :--- | :--- |
| **1. Intrinsic Syntactic Structure** | Sequence Perplexity & Motif Compression Ratio | Hyperparameter Sweep Loss | Evaluates sequence information content without reference to labels or kinematics. |
| **2. Extrinsic Pharmacological Shift** | Within-Subject Jensen-Shannon Divergence ($D_{\text{JS}}$) Pre vs Post Amph | Hyperparameter Sweep Loss | Measures sensitivity to a controlled experimental drug challenge. |
| **3. Kinematic Validation** | $\eta^2_{\text{speed}}$ & $\eta^2_{\text{dist}}$ | Secondary Sanity Check | Confirms physical grounding across manifold resolutions. |
| **4. Neural Discovery** | PETH $\Delta F/F$ Photometry Alignment at Transitions | Downstream Scientific Output | Explores dopamine modulation across WT vs DAT-KI without biasing model selection. |

---

## 4. Next Implementation Steps

1. **Implement Sequence Perplexity Metric**: Construct an entropy / perplexity evaluation script comparing within-vibe syllable transitions against shuffled controls.
2. **Implement Within-Session JSD Shift**: Compute $D_{\text{JS}}(P_{\text{pre}} \parallel P_{\text{post}})$ for all 30 sessions and evaluate the separation between Saline and Amphetamine groups.
3. **Re-rank Hyperparameter Trials**: Combine Syntactic Perplexity + Pharmacological JSD Shift into an objective score to select the final production model taxonomy.
